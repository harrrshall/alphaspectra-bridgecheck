from __future__ import annotations

from pathlib import Path

import yaml


PRODUCT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    PRODUCT_ROOT / "src" / "bandtrace" / "normative" / "bandtrace_v1.yaml"
)

EXPECTED_PLANTED_SCENARIOS = {
    "dropped_band",
    "edge_clamp",
    "reordered_bands",
    "id_ignored_metadata_sorted",
    "wavelength_nm_micron_mismatch",
    "missing_mandatory_fwhm",
    "invalid_present_srf",
    "routed_response_mismatch",
    "target_srf_outside_support",
    "claimed_wavelength_input_ignored",
    "undeclared_wavelength_input_used",
    "claimed_fwhm_input_ignored",
    "undeclared_fwhm_input_used",
    "routed_domain_outside_model_valid_range",
    "target_invariant_output_on_challenges",
    "stochastic_inference",
    "radiometric_quantity_mismatch",
    "undeclared_normalization",
    "undeclared_spatial_reduction",
    "context_dependent_undeclared_tap",
    "hidden_resampling_or_extrapolation",
    "duplicate_band_ids",
}


def _policy() -> dict[str, object]:
    return yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))


def test_release_matrix_is_the_frozen_twenty_two_scenario_contract() -> None:
    policy = _policy()
    matrix = policy["fault_matrix"]

    assert policy["frozen"] is True
    assert policy["policy_id"] == "bandtrace-0.1-r29"
    assert matrix["list_semantics"] == (
        "planted_release_faults_not_exhaustive_runtime_severity_enum"
    )
    assert matrix["planted_release_fault_entries_are_fixture_scenario_ids"] is True
    assert matrix["planted_scenario_id_need_not_equal_emitted_runtime_fault_code"] is True
    assert set(matrix["planted_release_faults"]) == EXPECTED_PLANTED_SCENARIOS
    assert len(matrix["planted_release_faults"]) == 22
    assert set(matrix["expected"]) == EXPECTED_PLANTED_SCENARIOS
    assert matrix["expected"]["id_ignored_metadata_sorted"] == {
        "axis": "route",
        "exit": 4,
        "emitted_fault": "reordered_bands",
        "forbidden_states": [
            "X2_DECLARED_TAP_MATCHES_ROUTE_ON_CHALLENGES",
            "X3_OUTPUT_DEPENDENCE_OBSERVED_ON_PROBES",
        ],
    }


def test_contract_preserves_orthogonal_state_axes() -> None:
    policy = _policy()
    requirements = policy["fault_matrix"]["release_requires"]

    assert requirements["preserve_unaffected_axis_evidence"] is True
    assert requirements[
        "every_completed_fault_fixture_exits_4_and_has_overall_conformance_fault"
    ] is True
    assert requirements["axis_specific_state_blocks"] == {
        "route_or_execution_fault": [
            "X2_DECLARED_TAP_MATCHES_ROUTE_ON_CHALLENGES",
            "X3_OUTPUT_DEPENDENCE_OBSERVED_ON_PROBES",
        ],
        "dependence_fault": ["X3_OUTPUT_DEPENDENCE_OBSERVED_ON_PROBES"],
        "spectral_support_fault": [
            "S2_APPROX_WITHIN_SUPPORT",
            "S3_SRF_WITHIN_DECLARED_SUPPORT",
        ],
    }
    assert policy["states"]["biological"] == [
        "T0_BIOLOGICAL_TRANSPORT_NOT_EVALUATED"
    ]


def test_frozen_canaries_outputs_limits_and_exit_codes_are_complete() -> None:
    policy = _policy()

    assert policy["canaries"] == [
        "C0_replay",
        "C1_declared_tap_agreement",
        "C2_value_dependence",
        "C3_wavelength_dependence",
        "C3_fwhm_dependence",
        "C4_order",
        "C5_target_neutral",
        "C6_edge_alias",
    ]
    assert set(policy["outputs"]["required"]) == {
        "report.json",
        "report.html",
        "route.csv",
        "canary_outputs.npz",
        "manifest.sha256",
    }
    assert policy["outputs"]["exit_codes"] == {
        "completed_no_conformance_fault": 0,
        "invalid_bundle": 2,
        "execution_failure": 3,
        "completed_with_conformance_fault": 4,
    }
    expected_limits = {
        "min_probes": 16,
        "max_probes": 256,
        "max_bands": 512,
        "max_spatial_cells_per_probe": 262144,
        "max_probe_file_bytes": 268435456,
        "max_expanded_float64_probe_bytes": 268435456,
        "max_canary_invocation_probe_bytes": 268435456,
        "max_adapter_output_bytes": 268435456,
        "subprocess_timeout_seconds": 120,
        "max_manifest_bytes": 1048576,
        "max_manifest_declared_files": 32,
        "max_manifest_declared_total_stat_bytes": 536870912,
        "manifest_file_count_and_total_stat_bytes_preflight_before_payload_read_or_hash": True,
        "max_npz_members": 32,
        "max_npz_uncompressed_bytes": 536870912,
        "max_npz_compression_ratio": 100.0,
        "max_subprocess_stdout_stderr_bytes": 1048576,
        "adapter_total_measured_seconds_failure_threshold": 600,
        "adapter_total_measured_seconds_failure_threshold_applies_to": [
            "numpy-linear-v1",
            "subprocess-npz-v1",
        ],
        "adapter_total_hard_deadline": False,
        "synchronous_parent_and_cleanup_preemptible": False,
        "subprocess_child_active_wall_poll": True,
        "external_hard_deadline_supervisor_required": True,
        "subprocess_cumulative_wall_scope": [
            "parent_request_serialization",
            "child_execution",
            "response_decode_and_validation",
            "invocation_cleanup",
        ],
        "max_adapter_invocations_formula": "2 * target_bands + 12",
        "max_c2_shift_cell_comparisons": 536870912,
        "c2_shift_cell_comparisons_formula": "(probe_count - 1) * probes.size",
        "max_cumulative_adapter_request_probe_bytes": 4294967296,
        "cumulative_adapter_request_probe_bytes_schedule": [
            "C0_three_full_baseline_requests",
            "C5_one_full_target_neutral_request",
            "C1_every_basis_chunk_request",
            "C1_one_rank4_spatial_request_when_applicable",
            "C2_one_full_request_per_target_band",
            "C3_six_full_metadata_requests",
            "C4_two_full_requests_when_target_bands_greater_than_one",
        ],
        "cumulative_adapter_request_probe_bytes_is_exact_sum_of_float64_request_probes_nbytes": True,
        "cumulative_adapter_request_probe_bytes_preflight_before_adapter_construction": True,
        "cumulative_request_byte_cap_is_not_peak_rss_guarantee": True,
        "catchable_memory_error_state": "invalid_bundle_or_execution_failure",
        "operating_system_oom_kill_cannot_be_converted_to_exit_code": True,
        "max_abs_declared_scalar_and_probe_value": 1000000000000.0,
        "max_abs_numpy_reference_artifact_numeric_value": 1000000000000.0,
        "reflectance_raw_value_min": -0.1,
        "reflectance_raw_value_max": 2.0,
        "min_normalization_scale": 0.01,
        "max_normalization_scale": 2.0,
        "normalization_offset_must_lie_inside_model_valid_range": True,
        "min_fwhm_nm": 1.0,
        "max_fwhm_nm": 50000.0,
        "min_spectral_wavelength_nm_after_unit_conversion": 100.0,
        "max_spectral_wavelength_nm_after_unit_conversion": 100000.0,
        "spectral_wavelength_bound_applies_to_centers_support_endpoints_and_srf_coordinates": True,
        "min_valid_range_width": 0.1,
        "max_abs_adapter_response_value": "1.0e150",
        "max_route_matrix_entries": 262144,
        "max_each_nonmanifest_output_bytes": 268435456,
        "max_all_five_output_bytes": 536870912,
        "every_derived_static_quantity_must_be_finite": True,
    }
    assert policy["limits"] == expected_limits


def test_security_publication_and_packaged_authority_terms_are_frozen() -> None:
    policy = _policy()
    security = policy["input_security"]
    outputs = policy["outputs"]

    assert security["npz_allow_pickle"] is False
    assert security["npz_allowed_zip_compression_methods"] == ["STORED", "DEFLATED"]
    assert security["subprocess_shell"] is False
    assert security["subprocess_new_process_group"] is True
    assert security[
        "subprocess_group_signal_attempt_after_every_successful_process_start"
    ] is True
    assert security["subprocess_leader_exit_observed_with_waitid_wnowait"] is True
    assert security["subprocess_group_signal_attempt_once_before_leader_reap"] is True
    assert security["subprocess_group_signal_delivery_guaranteed"] is False
    assert security["subprocess_group_signal_failure_can_leave_members_running"] is True
    assert security["subprocess_group_signal_failure_state"] == "execution_failure"
    assert security[
        "subprocess_direct_leader_kill_and_reap_fallback_configured_on_group_signal_error"
    ] is True
    assert security["subprocess_post_reap_numeric_pgid_signal_forbidden"] is True
    assert security["temporary_directory_mode"] == "0700"
    assert security["yaml_data_model"] == (
        "json_compatible_scalars_arrays_and_string_keyed_mappings_only"
    )
    assert security["max_structured_document_nesting_depth"] == 32
    assert security[
        "manifest_descriptor_held_and_revalidated_through_completed_parse"
    ] is True
    assert outputs["pinned_parent_directory_descriptor_opened_before_bundle_loading"]
    assert outputs[
        "staging_writes_cleanup_publication_and_fsync_are_relative_to_pinned_parent"
    ]
    assert outputs[
        "staging_path_identity_rechecked_against_open_descriptor_immediately_before_rename"
    ]
    assert outputs[
        "linux_libc_renameat2_availability_preflight_before_bundle_load_or_adapter_execution"
    ] is True
    assert outputs["atomic_directory_rename_into_destination"] == (
        "linux_renameat2_RENAME_NOREPLACE"
    )
    assert outputs["renameat2_unavailable_state"] == (
        "execution_failure_exit_3_without_unsafe_fallback"
    )
    assert outputs["concurrent_destination_is_never_replaced"] is True
    assert outputs["final_parent_path_and_published_inode_identity_checked"] is True
    assert outputs[
        "filesystem_threat_model_trusts_all_processes_running_as_bandtrace_unix_uid"
    ] is True
    assert outputs[
        "dedicated_account_and_private_output_parent_required_if_same_uid_cannot_be_trusted"
    ] is True
    assert outputs[
        "renameat2_cannot_condition_source_name_on_already_open_staging_inode"
    ] is True
    assert outputs[
        "successful_trusted_parent_publication_contains_all_fsynced_outputs_and_completion_manifest_at_atomic_rename"
    ] is True
    assert outputs[
        "postpublication_failure_does_not_rollback_or_delete_destination_path"
    ] is True
    assert outputs[
        "destination_existing_after_exit_3_is_untrusted_and_not_a_bandtrace_report"
    ] is True
    assert outputs[
        "same_uid_staging_source_name_substitution_may_leave_untrusted_destination_on_exit_3"
    ] is True
    assert outputs[
        "concurrent_postpublication_move_replace_mutate_or_remove_is_outside_claim"
    ] is True
    assert outputs["malicious_same_user_postpublication_defense_claimed"] is False
    assert outputs["prepublication_cleanup_is_nonrecursive"] is True
    assert outputs["prepublication_cleanup_refuses_unexpected_subdirectories"] is True
    assert outputs[
        "staging_name_identity_rechecked_immediately_before_rmdir"
    ] is True
    assert outputs["rmdir_cannot_condition_name_on_open_staging_inode"] is True
    assert outputs["malicious_same_uid_cleanup_name_swap_outside_guarantee"] is True
    assert outputs["no_postpublication_cleanup_attempted"] is True
    assert outputs["report_output_publication_contract"] == {
        "platform": "Linux",
        "mechanism": "renameat2(RENAME_NOREPLACE)",
        "fresh_destination_required": True,
        "private_parent_required": True,
        "source_identity_prechecked": True,
        "source_inode_conditioned_rename": False,
        "destination_identity_postchecked": True,
        "prepublication_cleanup_recursive": False,
        "prepublication_cleanup_source_identity_rechecked_before_rmdir": True,
        "same_uid_parent_name_race_fully_prevented": False,
        "postpublication_rollback": False,
    }
    assert outputs["categorical_comparisons_use_unrounded_binary64"] is True
    assert outputs["rounded_metric_and_threshold_may_display_equal_at_boundary"] is True
    assert outputs[
        "categorical_status_and_pass_boolean_are_authoritative_over_rounded_display"
    ] is True
    assert outputs["installed_source_digest_scope"] == "REGULAR_PY_FILES_ONLY"
    assert outputs["installed_source_digest_is_execution_attestation"] is False
    assert outputs[
        "imported_bytecode_interpreter_and_native_dependency_bytes_hashed"
    ] is False
    assert outputs["execution_environment_attested"] is False
    assert outputs["runtime_fingerprint_non_exhaustive"] is True
    assert outputs["installed_distribution_version_must_be_nonempty_string"] is True
    assert outputs["installed_distribution_metadata_failure_state"] == (
        "execution_failure_exit_3_before_bundle_loading"
    )
    assert outputs["packaged_normative_resources"] == {
        "product_document": "bandtrace/normative/BANDTRACE_PRODUCT.md",
        "machine_config": "bandtrace/normative/bandtrace_v1.yaml",
        "exact_bytes_must_match_compiled_sha256_constants_before_bundle_loading": True,
        "missing_unreadable_or_mismatched_state": (
            "execution_failure_exit_3_without_publication"
        ),
        "packaged_hash_gate_is_external_authentication": False,
        "coordinated_code_constants_and_resources_replacement_detected": False,
        "authenticity_requires_independently_trusted_distribution_digest_or_signature": True,
    }


def test_numeric_canary_and_spectral_terms_match_rev29_repairs() -> None:
    policy = _policy()
    construction = policy["canary_construction"]
    spectral = policy["spectral_support"]

    assert policy["determinism"]["known_answer"] == {
        "model_hash": "0" * 64,
        "sensor_hash": "1" * 64,
        "probe_hash": "2" * 64,
        "route_hash": "3" * 64,
        "policy_id": "bandtrace-0.1-r29",
        "base_seed_hex": "d9d2f0dc8aefdc56e2b3d91ca7112b698b7463f4f0a38747f7bd1b6c5e3257bf",
        "c4_target_bands": 5,
        "c4_shift": 1,
        "c4_permutation": [1, 2, 3, 4, 0],
        "c3_rank_target_ids": ["b0", "b1", "b2"],
        "c3_wavelength_subseed_hex": "378a64babd2f4e1452b4b136a3578160c7ab195d0671e9fed8d0a86ff72c27c8",
        "c3_wavelength_ranked_ids": ["b1", "b0", "b2"],
        "c3_wavelength_amplitudes_by_input_id": {"b0": 0.5, "b1": 0.25, "b2": 0.75},
        "c3_fwhm_subseed_hex": "1b0f69a72d1def6ff200738011f2cb8d2cd1fd3acf44e56cbe12ad5a9c10ee6f",
        "c3_fwhm_ranked_ids": ["b1", "b2", "b0"],
        "c3_fwhm_amplitudes_by_input_id": {"b0": 0.75, "b1": 0.25, "b2": 0.5},
    }
    assert construction["c1_rank4_spatial_challenge_rows"] == 4
    assert construction["c1_rank4_spatial_masks"] == [
        "complementary_even_odd_flattened_h_w_cells",
        "asymmetric_three_level_cycle_and_reverse",
    ]
    assert construction["c1_rank4_canonical_binary64_evaluation"] == (
        "float64_mean_each_target_band_over_h_w_then_declared_route_then_affine"
    )
    assert construction[
        "c1_single_spatial_cell_all_singleton_preserving_reducers_equivalent_is_reported"
    ] is True
    assert construction[
        "c1_two_spatial_cell_mean_median_midrange_equivalence_is_reported"
    ] is True
    assert construction["c4_id_only_request"] == (
        "hold_values_wavelength_and_fwhm_positions_fixed_and_permute_ids"
    )
    assert construction[
        "c4_id_only_expected_pre_core_change_must_be_strictly_greater_than"
    ] == 0.000001
    assert construction["c4_insufficient_id_binding_excitation_state"] == (
        "INCONCLUSIVE_INSUFFICIENT_ID_BINDING_EXCITATION"
    )
    assert construction["c4_pass_finding_requires_both_subtests"] == "ID_BOUND"
    assert construction["c6_invalid_srf_marks_only_its_target_id_unresolved"]
    assert construction[
        "c6_valid_outside_pairs_remain_reported_when_another_target_is_unresolved"
    ]
    assert spectral["interpolation"] == (
        "native_piecewise_linear_interval_sides_on_sorted_union_grid"
    )
    assert spectral["outside_supplied_domain_interval_side_values"] == "exact_zero"
    assert spectral["support_mass_integration"] == (
        "sum_whole_interval_areas_contained_in_support"
    )
    assert spectral["endpoint_to_distant_zero_outside_trapezoid_forbidden"] is True
    assert spectral["l1_method"] == "exact_absolute_linear_difference_per_interval"
    assert spectral["l1_sign_change_split"] == (
        "analytic_at_single_linear_difference_root"
    )
    assert spectral["maximum_endpoint_response_fraction_of_peak"] == 0.0001
    assert spectral["maximum_full_srf_unique_knots_per_channel"] == 250000
    assert spectral[
        "maximum_full_srf_interpolation_component_point_evaluations_per_audit"
    ] == 50000000


def test_claim_narrowing_and_route_audit_terms_remain_machine_explicit() -> None:
    policy = _policy()

    assert policy["states"]["executable"] == [
        "X0_NO_EXECUTABLE_OBSERVATION",
        "X1_REPLAY_STABLE_ON_PROBES",
        "X2_DECLARED_TAP_MATCHES_ROUTE_ON_CHALLENGES",
        "X3_OUTPUT_DEPENDENCE_OBSERVED_ON_PROBES",
    ]
    assert policy["radiometry"]["accepted_quantities"] == [
        "unitless_reflectance_factor"
    ]
    assert policy["radiometry"]["radiance_and_raw_dn_outside_v0_1"] is True
    assert policy["dependence"][
        "required_target_ids_must_be_explicit_nonempty_and_unique"
    ]
    assert policy["dependence"]["minimum_aggregate_absolute_route_weight"] == 0.0001
    assert policy["route"]["positive_route_weight_minimum"] == 0.000000000001
    assert policy["route"]["route_weight_domain"] == (
        "exactly_zero_or_at_least_positive_route_weight_minimum"
    )
    assert policy["outputs"]["route_audit_exact_fields"] == [
        "declared_weight_float64_hex",
        "declared_weight_is_strictly_positive",
        "declared_target_column_is_exactly_zero",
    ]
    assert policy["outputs"]["report_route_audit_representation"][
        "per_cell_object_list_forbidden"
    ] is True
    assert policy["outputs"]["html_is_bounded_summary_not_full_report_duplication"]
    assert policy["outputs"]["html_full_route_matrices_forbidden"] is True
