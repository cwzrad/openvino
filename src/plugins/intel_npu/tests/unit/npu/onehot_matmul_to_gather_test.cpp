// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "openvino/op/ops.hpp"
#include "openvino/runtime/core.hpp"

namespace {

std::shared_ptr<ov::Model> make_model(const int64_t on_value = 1) {
    using namespace ov;
    using namespace ov::op;

    auto indices = std::make_shared<v0::Parameter>(element::i64, PartialShape{1, Dimension::dynamic(), 2});
    indices->set_friendly_name("indices");
    auto depth = v0::Constant::create(element::i64, Shape{}, {7});
    auto on = v0::Constant::create(element::i64, Shape{}, {on_value});
    auto off = v0::Constant::create(element::i64, Shape{}, {0});
    auto one_hot = std::make_shared<v1::OneHot>(indices, depth, on, off, 3);
    auto one_hot_f32 = std::make_shared<v0::Convert>(one_hot, element::f32);
    auto one_hot_order = v0::Constant::create(element::i64, Shape{4}, {0, 2, 1, 3});
    auto transposed = std::make_shared<v1::Transpose>(one_hot_f32, one_hot_order);

    std::vector<int8_t> weights(2 * 7 * 3);
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = static_cast<int8_t>(static_cast<int>(i % 11) - 5);
    }
    auto weights_i8 = v0::Constant::create(element::i8, Shape{2, 7, 3}, weights);
    auto weights_f16 = std::make_shared<v0::Convert>(weights_i8, element::f16);
    auto scales = v0::Constant::create(element::f16, Shape{2, 1, 3}, std::vector<float>{0.125f, 0.25f, 0.5f,
                                                                                       0.0625f, 0.25f, 0.5f});
    auto scaled_weights = std::make_shared<v1::Multiply>(weights_f16, scales);
    auto table = std::make_shared<v0::Convert>(scaled_weights, element::f32);
    auto matmul = std::make_shared<v0::MatMul>(transposed, table, false, false);

    return std::make_shared<Model>(matmul, ParameterVector{indices});
}

ov::Tensor make_indices() {
    ov::Tensor indices(ov::element::i64, ov::Shape{1, 5, 2});
    const int64_t values[] = {0, 1, 6, 0, 7, 1, -1, 2, 3, 7};
    std::copy(std::begin(values), std::end(values), indices.data<int64_t>());
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

    ov::Tensor input(ov::element::i64, ov::Shape{2, 3, 2});
    const int64_t values[] = {0, 1, 6, 0, 7, 1, -1, 2, 3, 7, 4, 5};
    std::copy(std::begin(values), std::end(values), input.data<int64_t>());
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