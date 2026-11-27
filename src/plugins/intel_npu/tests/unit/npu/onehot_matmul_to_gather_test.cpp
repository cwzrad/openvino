// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "openvino/op/ops.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"

namespace {

std::shared_ptr<ov::Model> make_model(const int64_t on_value = 1,
                                      const ov::element::Type& index_type = ov::element::i32,
                                      bool widen_indices = false,
                                      const size_t table_depth = 7,
                                      float first_scale = 0.125f) {
    using namespace ov;
    using namespace ov::op;

    auto indices = std::make_shared<v0::Parameter>(index_type, PartialShape{1, Dimension::dynamic(), 2});
    indices->set_friendly_name("indices");
    auto depth = v0::Constant::create(element::i64, Shape{}, {table_depth});
    auto on = v0::Constant::create(element::i64, Shape{}, {on_value});
    auto off = v0::Constant::create(element::i64, Shape{}, {0});
    Output<Node> one_hot_indices = indices;
    if (widen_indices) {
        one_hot_indices = std::make_shared<v0::Convert>(indices, element::i64);
    }
    auto one_hot = std::make_shared<v1::OneHot>(one_hot_indices, depth, on, off, 3);
    auto one_hot_f32 = std::make_shared<v0::Convert>(one_hot, element::f32);
    auto one_hot_order = v0::Constant::create(element::i64, Shape{4}, {0, 2, 1, 3});
    auto transposed = std::make_shared<v1::Transpose>(one_hot_f32, one_hot_order);

    std::vector<int8_t> weights(2 * table_depth * 3);
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = static_cast<int8_t>(static_cast<int>(i % 11) - 5);
    }
    auto weights_i8 = v0::Constant::create(element::i8, Shape{2, table_depth, 3}, weights);
    auto weights_f16 = std::make_shared<v0::Convert>(weights_i8, element::f16);
    auto scales = v0::Constant::create(element::f16, Shape{2, 1, 3}, std::vector<float>{first_scale, 0.25f, 0.5f,
                                                                                       0.0625f, 0.25f, 0.5f});
    auto scaled_weights = std::make_shared<v1::Multiply>(weights_f16, scales);
    auto table = std::make_shared<v0::Convert>(scaled_weights, element::f32);
    auto matmul = std::make_shared<v0::MatMul>(transposed, table, false, false);

    return std::make_shared<Model>(matmul, ParameterVector{indices});
}

ov::Tensor make_indices() {
    ov::Tensor indices(ov::element::i32, ov::Shape{1, 5, 2});
    const int32_t values[] = {0, 1, 6, 0, 7, 1, -1, 2, 3, 7};
    std::copy(std::begin(values), std::end(values), indices.data<int32_t>());
    return indices;
}

}  // namespace

TEST(NpuOneHotMatMulToGather, GenericQuantizedTablePreservesResults) {
    ov::Core core;
    auto original = make_model();
    auto transformed = original->clone();

    EXPECT_TRUE(intel_npu::run_npu_graph_transformations(transformed));
    EXPECT_NO_THROW(transformed->validate_nodes_and_infer_types());

    size_t gather_count = 0;
    size_t matmul_count = 0;
    for (const auto& node : transformed->get_ordered_ops()) {
        gather_count += node->get_type_info() == ov::op::v8::Gather::get_type_info_static();
        matmul_count += node->get_type_info() == ov::op::v0::MatMul::get_type_info_static();
        EXPECT_NE(node->get_type_info(), ov::op::v1::Maximum::get_type_info_static());
        EXPECT_NE(node->get_type_info(), ov::op::v1::Minimum::get_type_info_static());
    }
    EXPECT_EQ(gather_count, 1);
    EXPECT_EQ(matmul_count, 0);

    auto original_compiled = core.compile_model(original, "CPU");
    auto transformed_compiled = core.compile_model(transformed, "CPU");
    const auto input = make_indices();
    auto original_request = original_compiled.create_infer_request();
    auto transformed_request = transformed_compiled.create_infer_request();
    original_request.set_input_tensor(input);
    transformed_request.set_input_tensor(input);
    original_request.infer();
    transformed_request.infer();
    const auto expected = original_request.get_output_tensor();
    const auto actual = transformed_request.get_output_tensor();

    ASSERT_EQ(actual.get_shape(), expected.get_shape());
    const auto* expected_data = expected.data<const float>();
    const auto* actual_data = actual.data<const float>();
    for (size_t i = 0; i < expected.get_size(); ++i) {
        EXPECT_NEAR(actual_data[i], expected_data[i], 1e-6f) << "Mismatch at output index " << i;
    }
    EXPECT_FALSE(intel_npu::run_npu_graph_transformations(transformed));
}

TEST(NpuOneHotMatMulToGather, SkipsNonUnitOnValue) {
    auto model = make_model(2);
    EXPECT_FALSE(intel_npu::run_npu_graph_transformations(model));
}

TEST(NpuOneHotMatMulToGather, PreservesMultipleBatches) {
    ov::Core core;
    auto original = make_model();
    original->reshape(ov::PartialShape{2, ov::Dimension::dynamic(), 2});
    auto transformed = original->clone();
    ASSERT_TRUE(intel_npu::run_npu_graph_transformations(transformed));

    ov::Tensor input(ov::element::i32, ov::Shape{2, 3, 2});
    const int32_t values[] = {0, 1, 6, 0, 7, 1, -1, 2, 3, 7, 4, 5};
    std::copy(std::begin(values), std::end(values), input.data<int32_t>());
    auto original_request = core.compile_model(original, "CPU").create_infer_request();
    auto transformed_request = core.compile_model(transformed, "CPU").create_infer_request();
    original_request.set_input_tensor(input);
    transformed_request.set_input_tensor(input);
    original_request.infer();
    transformed_request.infer();
    const auto expected = original_request.get_output_tensor();
    const auto actual = transformed_request.get_output_tensor();
    ASSERT_EQ(actual.get_shape(), expected.get_shape());
    const auto* expected_data = expected.data<const float>();
    const auto* actual_data = actual.data<const float>();
    for (size_t i = 0; i < expected.get_size(); ++i) {
        EXPECT_NEAR(actual_data[i], expected_data[i], 1e-6f) << "Mismatch at output index " << i;
    }
}

TEST(NpuOneHotMatMulToGather, PreservesI64IndicesAndInvalidExtremes) {
    auto original = make_model(1, ov::element::i64, false, 10240);
    auto transformed = original->clone();
    ASSERT_TRUE(intel_npu::run_npu_graph_transformations(transformed));
    for (const auto& node : transformed->get_ordered_ops()) {
        if (node->get_type_info() == ov::op::v8::Gather::get_type_info_static()) {
            EXPECT_EQ(node->get_input_element_type(1), ov::element::i64);
        }
    }
    ov::Tensor input(ov::element::i64, ov::Shape{1, 8, 2});
    const int64_t values[] = {std::numeric_limits<int64_t>::min(), -4294967296LL, -1, 0, 2048, 2049, 10238, 10239,
                              10240, 10241, 32767, 32768, 4294967296LL, 4294967303LL, 9007199254740999LL,
                              std::numeric_limits<int64_t>::max()};
    std::copy(std::begin(values), std::end(values), input.data<int64_t>());
    ov::TensorVector expected{ov::Tensor()};
    ov::TensorVector actual{ov::Tensor()};
    ASSERT_TRUE(original->evaluate(expected, {input}));
    ASSERT_TRUE(transformed->evaluate(actual, {input}));
    ASSERT_EQ(actual[0].get_shape(), expected[0].get_shape());
    for (size_t i = 0; i < expected[0].get_size(); ++i) {
        EXPECT_FLOAT_EQ(actual[0].data<const float>()[i], expected[0].data<const float>()[i]) << i;
    }
}

TEST(NpuOneHotMatMulToGather, SkipsI64DepthBeyondGuardLimit) {
    auto supported = make_model(1, ov::element::i64, false, 16384);
    EXPECT_TRUE(intel_npu::run_npu_graph_transformations(supported));
    auto model = make_model(1, ov::element::i64, false, 16385);
    EXPECT_FALSE(intel_npu::run_npu_graph_transformations(model));
}

TEST(NpuOneHotMatMulToGather, HonorsTransformationCallback) {
    auto model = make_model();
    ov::pass::Manager manager;
    manager.register_pass<intel_npu::pass::OneHotMatMulToGather>();
    manager.get_pass_config()->set_callback<intel_npu::pass::OneHotMatMulToGather>(
        [](const std::shared_ptr<const ov::Node>&) { return true; });
    EXPECT_FALSE(manager.run_passes(model));
}

TEST(NpuOneHotMatMulToGather, SkipsNonFiniteOrOverflowingScale) {
    for (const float scale : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(), 60000.0f}) {
        auto model = make_model(1, ov::element::i32, false, 7, scale);
        EXPECT_FALSE(intel_npu::run_npu_graph_transformations(model)) << scale;
    }
}

TEST(NpuOneHotMatMulToGather, AcceptsI32WidenedToI64) {
    ov::Core core;
    auto original = make_model(1, ov::element::i32, true);
    auto transformed = original->clone();
    ASSERT_TRUE(intel_npu::run_npu_graph_transformations(transformed));
    for (const auto& node : transformed->get_ordered_ops()) {
        if (node->get_type_info() == ov::op::v8::Gather::get_type_info_static()) {
            EXPECT_EQ(node->get_input_element_type(1), ov::element::i32);
        }
    }
    ov::Tensor input(ov::element::i32, ov::Shape{1, 4, 2});
    const int32_t values[] = {std::numeric_limits<int32_t>::min(), -1, 0, 6, 7, 8,
                              std::numeric_limits<int32_t>::max(), 1};
    std::copy(std::begin(values), std::end(values), input.data<int32_t>());
    auto original_request = core.compile_model(original, "CPU").create_infer_request();
    auto transformed_request = core.compile_model(transformed, "CPU").create_infer_request();
    original_request.set_input_tensor(input);
    transformed_request.set_input_tensor(input);
    original_request.infer();
    transformed_request.infer();
    const auto expected = original_request.get_output_tensor();
    const auto actual = transformed_request.get_output_tensor();
    ASSERT_EQ(actual.get_shape(), expected.get_shape());
    for (size_t i = 0; i < expected.get_size(); ++i) {
        EXPECT_NEAR(actual.data<const float>()[i], expected.data<const float>()[i], 1e-6f) << i;
    }
}

TEST(NpuOneHotMatMulToGather, HandlesLargeTableAndInvalidI32Indices) {
    ov::Core core;
    auto original = make_model(1, ov::element::i32, false, 100000);
    auto transformed = original->clone();
    ASSERT_TRUE(intel_npu::run_npu_graph_transformations(transformed));
    ov::Tensor input(ov::element::i32, ov::Shape{1, 5, 2});
    const int32_t values[] = {2049, 10239, 65504, 65505, 65535, 99999, 100000, -1,
                              std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()};
    std::copy(std::begin(values), std::end(values), input.data<int32_t>());
    auto original_request = core.compile_model(original, "CPU").create_infer_request();
    auto transformed_request = core.compile_model(transformed, "CPU").create_infer_request();
    original_request.set_input_tensor(input);
    transformed_request.set_input_tensor(input);
    original_request.infer();
    transformed_request.infer();
    const auto expected = original_request.get_output_tensor();
    const auto actual = transformed_request.get_output_tensor();
    ASSERT_EQ(actual.get_shape(), expected.get_shape());
    for (size_t i = 0; i < expected.get_size(); ++i) {
        EXPECT_NEAR(actual.data<const float>()[i], expected.data<const float>()[i], 1e-6f) << i;
    }
}
