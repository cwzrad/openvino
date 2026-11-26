// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <sstream>
#include <typeinfo>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "intel_npu/utils/utils.hpp"
#include "openvino/core/layout.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#define MATCHER_SCOPE(region) const std::string matcher_name = std::string(typeid(*this).name()) + "_" #region

namespace intel_npu {

namespace {

std::optional<int64_t> get_integer_scalar(const ov::Output<ov::Node>& value) {
    const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(value.get_node_shared_ptr());
    if (!constant || !constant->get_shape().empty()) {
        return std::nullopt;
    }
    if (constant->get_element_type() == ov::element::i64) {
        return constant->cast_vector<int64_t>().front();
    }
    if (constant->get_element_type() == ov::element::i32) {
        return constant->cast_vector<int32_t>().front();
    }
    return std::nullopt;
}

std::optional<std::vector<int64_t>> get_integer_vector(const ov::Output<ov::Node>& value) {
    const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(value.get_node_shared_ptr());
    if (!constant) {
        return std::nullopt;
    }
    if (constant->get_element_type() == ov::element::i64) {
        return constant->cast_vector<int64_t>();
    }
    if (constant->get_element_type() == ov::element::i32) {
        const auto values = constant->cast_vector<int32_t>();
        return std::vector<int64_t>(values.begin(), values.end());
    }
    return std::nullopt;
}

bool is_integer_index_type(const ov::element::Type& type) {
    return type == ov::element::i32 || type == ov::element::i64;
}

}  // namespace

namespace pass {

OneHotMatMulToGather::OneHotMatMulToGather() {
    MATCHER_SCOPE(OneHotMatMulToGather);

    using namespace ov::pass::pattern;
    const auto one_hot = wrap_type<ov::op::v1::OneHot>();
    const auto one_hot_convert = wrap_type<ov::op::v0::Convert>({one_hot});
    const auto transpose = wrap_type<ov::op::v1::Transpose>({one_hot_convert, any_input()});
    const auto matmul = wrap_type<ov::op::v0::MatMul>({transpose, any_input()});

    const ov::matcher_pass_callback callback = [one_hot, one_hot_convert, transpose](ov::pass::pattern::Matcher& matcher) {
        const auto& pattern_map = matcher.get_pattern_value_map();
        const auto one_hot_node = ov::as_type_ptr<ov::op::v1::OneHot>(pattern_map.at(one_hot).get_node_shared_ptr());
        const auto one_hot_convert_node =
            ov::as_type_ptr<ov::op::v0::Convert>(pattern_map.at(one_hot_convert).get_node_shared_ptr());
        const auto transpose_node =
            ov::as_type_ptr<ov::op::v1::Transpose>(pattern_map.at(transpose).get_node_shared_ptr());
        const auto matmul_node = ov::as_type_ptr<ov::op::v0::MatMul>(matcher.get_match_root());

        if (one_hot_node->get_axis() != 3 || matmul_node->get_transpose_a() || matmul_node->get_transpose_b()) {
            return false;
        }

        const auto indices = one_hot_node->input_value(0);
        const auto index_shape = indices.get_partial_shape();
        if (!is_integer_index_type(indices.get_element_type()) || !index_shape.rank().is_static() ||
            index_shape.rank().get_length() != 3 || !index_shape[2].is_static()) {
            return false;
        }

        const auto depth = get_integer_scalar(one_hot_node->input_value(1));
        const auto on_value = get_integer_scalar(one_hot_node->input_value(2));
        const auto off_value = get_integer_scalar(one_hot_node->input_value(3));
        if (!depth || !on_value || !off_value || *depth <= 0 || *on_value != 1 || *off_value != 0) {
            return false;
        }

        const auto permutation = get_integer_vector(transpose_node->input_value(1));
        if (!permutation || *permutation != std::vector<int64_t>{0, 2, 1, 3}) {
            return false;
        }

        const auto table_convert =
            ov::as_type_ptr<ov::op::v0::Convert>(matmul_node->input_value(1).get_node_shared_ptr());
        if (!table_convert) {
            return false;
        }
        const auto multiply =
            ov::as_type_ptr<ov::op::v1::Multiply>(table_convert->input_value(0).get_node_shared_ptr());
        if (!multiply) {
            return false;
        }

        ov::Output<ov::Node> scale_output;
        auto weight_convert = ov::as_type_ptr<ov::op::v0::Convert>(multiply->input_value(0).get_node_shared_ptr());
        auto scale = ov::as_type_ptr<ov::op::v0::Constant>(multiply->input_value(1).get_node_shared_ptr());
        if (!weight_convert || !scale) {
            weight_convert = ov::as_type_ptr<ov::op::v0::Convert>(multiply->input_value(1).get_node_shared_ptr());
            scale = ov::as_type_ptr<ov::op::v0::Constant>(multiply->input_value(0).get_node_shared_ptr());
        }
        if (!weight_convert || !scale ||
            !ov::as_type_ptr<ov::op::v0::Constant>(weight_convert->input_value(0).get_node_shared_ptr())) {
            return false;
        }
        scale_output = scale->output(0);

        const auto weights = ov::as_type_ptr<ov::op::v0::Constant>(weight_convert->input_value(0).get_node_shared_ptr());
        const auto weight_shape = weights->get_shape();
        const auto scale_shape = scale->get_shape();
        const auto coordinate_count = static_cast<size_t>(index_shape[2].get_length());
        if ((weights->get_element_type() != ov::element::i8 && weights->get_element_type() != ov::element::u8) ||
            weight_shape.size() != 3 || scale_shape.size() != 3 || weight_shape[0] != coordinate_count ||
            weight_shape[1] != static_cast<size_t>(*depth) || weight_shape[2] == 0 || scale_shape[0] != weight_shape[0] ||
            scale_shape[1] != 1 || scale_shape[2] != weight_shape[2] ||
            weight_convert->get_destination_type() != scale->get_element_type() ||
            table_convert->get_destination_type() != matmul_node->get_input_element_type(1) ||
            one_hot_convert_node->get_destination_type() != matmul_node->get_input_element_type(0)) {
            return false;
        }

        const auto index_type = indices.get_element_type();
        const auto zero_index = ov::op::v0::Constant::create(index_type, ov::Shape{}, {0});
        const auto max_index = ov::op::v0::Constant::create(index_type, ov::Shape{}, {*depth - 1});
        const auto depth_index = ov::op::v0::Constant::create(index_type, ov::Shape{}, {*depth});
        const auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        const auto gather_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {2, 0, 1});
        const auto output_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {1, 0, 2, 3});
        const auto mask_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
        const auto unsqueeze_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        const auto mask_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {3});

        const auto non_negative = std::make_shared<ov::op::v1::GreaterEqual>(indices, zero_index);
        const auto below_depth = std::make_shared<ov::op::v1::Less>(indices, depth_index);
        const auto valid_indices = std::make_shared<ov::op::v1::LogicalAnd>(non_negative, below_depth);
        const auto clamped_low = std::make_shared<ov::op::v1::Maximum>(indices, zero_index);
        const auto clamped_indices = std::make_shared<ov::op::v1::Minimum>(clamped_low, max_index);
        const auto gather_indices = std::make_shared<ov::op::v1::Transpose>(clamped_indices, gather_order);
        const auto gathered_weights =
            std::make_shared<ov::op::v8::Gather>(weights, gather_indices, gather_axis, 1);
        const auto gathered_converted = std::make_shared<ov::op::v0::Convert>(gathered_weights, weight_convert->get_destination_type());
        const auto expanded_scale = std::make_shared<ov::op::v0::Unsqueeze>(scale_output, unsqueeze_axis);
        const auto dequantized = std::make_shared<ov::op::v1::Multiply>(gathered_converted, expanded_scale);
        const auto table_type_values = std::make_shared<ov::op::v0::Convert>(dequantized, table_convert->get_destination_type());
        const auto gathered = std::make_shared<ov::op::v1::Transpose>(table_type_values, output_order);
        const auto mask_transpose = std::make_shared<ov::op::v1::Transpose>(valid_indices, mask_order);
        const auto expanded_mask = std::make_shared<ov::op::v0::Unsqueeze>(mask_transpose, mask_axis);
        const auto zero_value = ov::op::v0::Constant::create(table_convert->get_destination_type(), ov::Shape{}, {0.0f});
        const auto masked = std::make_shared<ov::op::v1::Select>(expanded_mask, gathered, zero_value);

        ov::NodeVector new_nodes;
        const auto add_new_node = [&new_nodes](const auto& node) {
            new_nodes.push_back(node);
        };
        add_new_node(zero_index);
        add_new_node(max_index);
        add_new_node(depth_index);
        add_new_node(gather_axis);
        add_new_node(gather_order);
        add_new_node(output_order);
        add_new_node(mask_order);
        add_new_node(unsqueeze_axis);
        add_new_node(mask_axis);
        add_new_node(non_negative);
        add_new_node(below_depth);
        add_new_node(valid_indices);
        add_new_node(clamped_low);
        add_new_node(clamped_indices);
        add_new_node(gather_indices);
        add_new_node(gathered_weights);
        add_new_node(gathered_converted);
        add_new_node(expanded_scale);
        add_new_node(dequantized);
        add_new_node(table_type_values);
        add_new_node(gathered);
        add_new_node(mask_transpose);
        add_new_node(expanded_mask);
        add_new_node(zero_value);
        add_new_node(masked);

        ov::NodeVector source_nodes;
        source_nodes.push_back(one_hot_node);
        source_nodes.push_back(one_hot_convert_node);
        source_nodes.push_back(transpose_node);
        source_nodes.push_back(matmul_node);
        source_nodes.push_back(table_convert);
        source_nodes.push_back(multiply);
        source_nodes.push_back(weight_convert);
        source_nodes.push_back(weights);
        source_nodes.push_back(scale);
        ov::copy_runtime_info(source_nodes, new_nodes);
        masked->set_friendly_name(matmul_node->get_friendly_name());
        ov::replace_node(matmul_node, masked);
        return true;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(matmul, matcher_name), callback);
}

}  // namespace pass

bool run_npu_graph_transformations(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<pass::OneHotMatMulToGather>();
    return manager.run_passes(model);
}

bool should_use_host_compile_interpreter(const std::shared_ptr<const ov::Model>& model,
                                         ov::intel_npu::CompilerType compilerType,
                                         bool compilationModeSet,
                                         bool dynamicShapeToStatic) {
    if (compilerType != ov::intel_npu::CompilerType::PLUGIN || compilationModeSet || dynamicShapeToStatic) {
        return false;
    }

    // HostCompile supports dynamic dimensions only; a single dynamic-rank input rules the model out.
    const auto hasStaticRank = [](const auto& port) {
        return port.get_partial_shape().rank().is_static();
    };

    // Any dynamic dimension other than batch, bounded or unbounded, makes the model a HostCompile model: HostCompile
    // sizes dynamic buffers at run time from the bound tensor or the predicted output shape. A dynamic batch alone is
    // not supported well by HostCompile yet, so such models are left to plugin batching.
    const auto isDynamicPort = [](const auto& port) {
        return batch_helpers::hasOtherDynamicDims(port.get_partial_shape());
    };

    // Only the inputs decide: shape inference does not reach the outputs of some models, which then look fully
    // dynamic even though the model is a HostCompile one.
    const auto& modelInputs = model->inputs();
    if (!std::all_of(modelInputs.begin(), modelInputs.end(), hasStaticRank)) {
        return false;
    }

    return std::any_of(modelInputs.begin(), modelInputs.end(), isDynamicPort);
}

namespace batch_helpers {

bool hasOtherDynamicDims(const ov::PartialShape& shape) {
    for (size_t dim_idx = 1; dim_idx < shape.size(); dim_idx++) {
        if (shape[dim_idx].is_dynamic()) {
            return true;  // Found dynamic dimension other than batch
        }
    }
    return false;
}

bool checkModelDynamicDims(const std::shared_ptr<const ov::Model>& model) {
    // Check parameters (inputs)
    const auto& params = model->get_parameters();
    for (const auto& param : params) {
        const auto& shape = param->get_partial_shape();
        if (hasOtherDynamicDims(shape)) {
            return true;
        }
    }

    // Check results (outputs)
    const auto& results = model->get_results();
    for (const auto& result : results) {
        const auto& shape = result->get_output_partial_shape(0);
        if (hasOtherDynamicDims(shape)) {
            return true;
        }
    }

    return false;
}

bool validateModelBatch(const std::shared_ptr<const ov::Model>& model, Logger logger) {
    std::set<ov::Output<const ov::Node>> batchedInputs;
    std::set<ov::Output<const ov::Node>> batchedOutputs;
    std::set<size_t> sBatchSize;

    // Limitation: Plugin batching is not supported when there are dynamic
    // dimensions other than the batch dimension.
    if (checkModelDynamicDims(model) && model->is_dynamic()) {
        return false;
    }

    const auto& params = model->get_parameters();
    for (size_t input_id = 0; input_id < params.size(); input_id++) {
        const auto& input = params[input_id];
        const auto& shape = input->get_partial_shape();
        ov::Layout layout = ov::layout::get_layout(input);

        // Batching on plugin is working only when batching is found on 0th dimension
        if ((shape.size() &&
             shape[intel_npu::utils::BATCH_AXIS].get_max_length() != intel_npu::utils::DEFAULT_BATCH_SIZE) ||
            (ov::layout::has_batch(layout) && ov::layout::batch_idx(layout) == intel_npu::utils::BATCH_AXIS)) {
            const auto& staticShape = shape.is_dynamic() ? shape.get_max_shape() : input->get_shape();
            batchedInputs.insert(params[input_id]->output(0));

            if (shape.rank().is_dynamic()) {
                OPENVINO_THROW("Shapes with dynamic rank are not supported.");
            } else {
                sBatchSize.insert(staticShape[intel_npu::utils::BATCH_AXIS]);
            }
        } else {
            // gather some diagnostic info
            std::optional<size_t> batch_dim_index_detected;
            for (size_t i = 1; i < shape.size(); i++) {
                if (shape[i].has_symbol()) {
                    batch_dim_index_detected = i;
                    break;
                }
            }
            std::stringstream sstream;
            sstream << "Only networks with inputs batched by 0th dimension are supported. ";
            if (batch_dim_index_detected.has_value()) {
                sstream << "The batch has been detected on: " << batch_dim_index_detected.value()
                        << " dimension instead. ";
            } else {
                sstream << "The batch hasn't been detected at all. ";
            }
            sstream << "Please check input id: " << input_id << " by the name: " << input->get_friendly_name()
                    << ", layout: " << layout.to_string() << ", is_dynamic: " << shape.is_dynamic();
            logger.info("%s", sstream.str().c_str());
            return false;
        }
    }
    for (const auto& output : model->get_results()) {
        const auto& shape = output->get_output_partial_shape(0);
        ov::Layout layout = ov::layout::get_layout(output);

        // Batching on plugin is working only when batching is found on 0th dimension
        if ((shape.size() &&
             shape[intel_npu::utils::BATCH_AXIS].get_max_length() != intel_npu::utils::DEFAULT_BATCH_SIZE) ||
            (ov::layout::has_batch(layout) && ov::layout::batch_idx(layout) == intel_npu::utils::BATCH_AXIS)) {
            const auto& node = output->input_value(0);
            const auto& staticShape = shape.is_dynamic() ? shape.get_max_shape() : output->get_shape();
            batchedOutputs.insert(ov::Output<const ov::Node>(node.get_node(), node.get_index()));

            if (shape.rank().is_dynamic()) {
                OPENVINO_THROW("Shapes with dynamic rank are not supported.");
            } else {
                sBatchSize.insert(staticShape[intel_npu::utils::BATCH_AXIS]);
            }
        } else {
            logger.info("Only networks with outputs batched by 0th dimension are supported. Please check an output by "
                        "the name: %s, layout: %s",
                        output->get_friendly_name().c_str(),
                        layout.to_string().c_str());
            return false;
        }
    }
    if (!batchedInputs.size() || !batchedOutputs.size()) {
        logger.info(
            "Only networks with inputs/outputs featuring batched dim are supported! Got inputs: %ld, outputs: %ld",
            batchedInputs.size(),
            batchedOutputs.size());
        return false;
    }

    if (sBatchSize.size() != 1) {
        logger.info("Batching size shall have same value for all tensors! Got unique batch sizes number: %ld",
                    sBatchSize.size());
        return false;
    }

    if (*sBatchSize.begin() == intel_npu::utils::DEFAULT_BATCH_SIZE) {
        logger.info("PLUGIN batch won't be applied, got default batch value : %ld", *sBatchSize.begin());
        return false;
    }

    auto node_info_printer = [&logger](const auto& ov_node, std::string nodeType) {
        logger.info("%s: %s has shape value: %s",
                    nodeType.c_str(),
                    ov_node.get_any_name().c_str(),
                    ov_node.get_partial_shape().to_string().c_str());
    };

    for (const auto& ov_node : batchedInputs) {
        node_info_printer(ov_node, "Input");
    }
    for (const auto& ov_node : batchedOutputs) {
        node_info_printer(ov_node, "Output");
    }

    return true;
}

bool deBatchModel(std::shared_ptr<ov::Model>& model,
                  ov::Dimension newBatch,
                  std::optional<ov::Dimension>& originalBatch) {
    try {
        std::map<std::string, ov::PartialShape> newShapes;
        auto shapeChanged = false;
        for (auto&& item : model->get_parameters()) {
            auto layout = item->get_layout();
            auto partShape = item->get_partial_shape();
            if (ov::layout::has_batch(layout)) {
                shapeChanged = true;
                originalBatch = partShape[ov::layout::batch_idx(layout)];
                partShape[ov::layout::batch_idx(layout)] = newBatch;
            }
            newShapes.emplace(item->get_friendly_name(), partShape);
        }
        model->reshape(newShapes);
        return shapeChanged;
    } catch (const std::exception&) {
        // Don't throw - let caller handle the failure
        return false;
    }
}

std::tuple<std::shared_ptr<ov::Model>, bool> handlePluginBatching(
    std::shared_ptr<const ov::Model> model,
    const std::function<void(ov::intel_npu::BatchMode)>& updateBatchMode,
    std::optional<ov::intel_npu::BatchMode> batchMode,
    std::optional<ov::Dimension>& originalBatch,
    Logger logger) {
    // Keep the original model for all no-op/early-return paths.
    // A mutable clone is created only when plugin batching is actually about to be applied.
    auto resultModel = std::const_pointer_cast<ov::Model>(model);
    auto successfullyDebatched = false;

    auto batchModeIsAvailable = batchMode.has_value();
    ov::intel_npu::BatchMode effectiveBatchMode =
        batchModeIsAvailable ? batchMode.value() : ov::intel_npu::BatchMode::AUTO;

    if (batchModeIsAvailable) {
        const auto isAutoOrPluginBatch = (effectiveBatchMode == ov::intel_npu::BatchMode::PLUGIN ||
                                          effectiveBatchMode == ov::intel_npu::BatchMode::AUTO);

        if (!isAutoOrPluginBatch) {
            return {resultModel, successfullyDebatched};
        }
    }

    try {
        const auto pluginBatchingIsSupported = validateModelBatch(model, logger);

        if (!pluginBatchingIsSupported) {
            if (batchModeIsAvailable && effectiveBatchMode == ov::intel_npu::BatchMode::AUTO) {
                logger.info("Batching will be handled by compiler.");
                updateBatchMode(ov::intel_npu::BatchMode::COMPILER);
            }
            return {resultModel, successfullyDebatched};
        }

        logger.info("Attempting to handle batching on the plugin side.");
        // Clone right before mutation to avoid extra memory usage when batching is skipped.
        resultModel = model->clone();

        try {
            originalBatch = ov::get_batch(resultModel);
            ov::set_batch(resultModel, ov::Dimension(1));
            successfullyDebatched = true;
        } catch (const std::exception& ex) {
            logger.warning("The plugin couldn't resize a batched model due to exception: %s.\n"
                           "Trying to debatch it...",
                           ex.what());

            if (!deBatchModel(resultModel, ov::Dimension(1), originalBatch)) {
                OPENVINO_THROW("Cannot debatch a model");
            }
            logger.info("The model has been debatched successfully");
            successfullyDebatched = true;
        }
        if (batchModeIsAvailable) {
            // If we have successfully debatched the model on the PLUGIN side, we should
            // avoid repeating the same in the compiler by resetting the batch mode
            logger.info("The model was reshaped to batch size 1 on the plugin side.");
            updateBatchMode(ov::intel_npu::BatchMode::COMPILER);
        }
    } catch (const std::exception& ex) {
        // If plugin-side transformation failed, keep the original model and drop the clone
        resultModel = std::const_pointer_cast<ov::Model>(model);
        if (effectiveBatchMode == ov::intel_npu::BatchMode::AUTO) {
            logger.info("Couldn't validate and reshape the model. Batching will be handled by compiler. Error: %s",
                        ex.what());
            if (batchModeIsAvailable) {
                // If we failed to handle batching on the plugin side, we should reset the batch mode to default
                // COMPILER but only if the batch mode is available, otherwise we might be running on an older compiler
                // which doesn't support batch mode at all
                updateBatchMode(ov::intel_npu::BatchMode::COMPILER);
            }
        } else {
            OPENVINO_THROW("Couldn't validate and reshape the model for PLUGIN batch mode. Error: ", ex.what());
        }
    }

    return {resultModel, successfullyDebatched};
}

}  // namespace batch_helpers
}  // namespace intel_npu
