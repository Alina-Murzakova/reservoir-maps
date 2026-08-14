import logging

import numpy as np
import pandas as pd

from .one_phase_model import get_sw


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


ADAPTED_OFP_COLUMNS = [
    "well_number",
    "Fw",
    "m1",
    "Fo",
    "m2",
    "Swc",
    "Sor",
    "adaptation_stage",
    "eps_so_min",
    "eps_so_mean",
    "eps_so_max",
    "forced_points",
]


def calculate_weight_multipliers(
    So_init_wells,
    So_current_wells,
    Sor,
    alpha,
    min_multiplier,
):
    """
    Reduce the influence of points whose current saturation exceeds the initial value.

    Args:
        So_init_wells: Initial oil saturation at well trajectory points.
        So_current_wells: Current oil saturation at well trajectory points.
        Sor: Residual oil saturation.
        alpha: Exponential decay coefficient applied to inconsistency severity.
        min_multiplier: Lower bound for the resulting influence multiplier.

    Returns:
        tuple:
            np.ndarray: Influence multiplier for every well trajectory point.
            np.ndarray: Boolean mask of points where So_current > So_init.
    """
    So_init_wells = np.asarray(So_init_wells, dtype="float64")
    So_current_wells = np.asarray(So_current_wells, dtype="float64")
    bad_mask = So_current_wells > So_init_wells

    denominator = np.maximum(So_init_wells - Sor, np.finfo("float64").eps)
    severity = np.zeros_like(So_init_wells)
    severity[bad_mask] = (So_current_wells[bad_mask] - So_init_wells[bad_mask]) / denominator[bad_mask]

    multipliers = np.exp(-alpha * severity)
    multipliers = np.clip(multipliers, min_multiplier, 1.0).astype("float32")
    lowered_mask = multipliers < 1.0
    logger.info(
        "Reduced influence weights for %d of %d well trajectory points (minimum multiplier %.3f)",
        int(np.count_nonzero(lowered_mask)),
        len(multipliers),
        float(multipliers.min(initial=1.0)),
    )
    return multipliers, bad_mask


def _production_normalization(data_wells, minimum):
    """
    Normalize cumulative well production for the eps_so calculation.

    Args:
        data_wells: Well data containing work markers and cumulative oil production.
        minimum: Lower bound for the normalized cumulative production.

    Returns:
        tuple:
            dict: Normalized cumulative production indexed by well number.
            float: Production normalization base or NaN when the fallback is used.
    """
    production = data_wells.loc[
        (data_wells["work_marker"] == "prod")
        & np.isfinite(data_wells["Qo_cumsum"])
        & (data_wells["Qo_cumsum"] > 0),
        "Qo_cumsum",
    ].astype(float)

    if len(production) < 5:
        normalized = {well: 0.25 for well in data_wells["well_number"]}
        base = np.nan
    else:
        base = float(production.max() if len(production) < 20 else np.nanpercentile(production, 95))
        normalized = {
            row["well_number"]: float(row["Qo_cumsum"]) / base
            for _, row in data_wells.iterrows()
        }

    return {well: float(np.clip(value, minimum, 1.0)) for well, value in normalized.items()}, base


def _store_point_arrays(data_wells, So_current_wells, eps_so_wells, weight_multipliers):
    """
    Store trajectory-aligned diagnostics without changing the input/output well schema.

    Args:
        data_wells: Well table whose rows contain trajectory point arrays.
        So_current_wells: Current oil saturation at all trajectory points.
        eps_so_wells: Calculated eps_so at all trajectory points.
        weight_multipliers: Influence multipliers at all trajectory points.
    """
    offset = 0
    for index, row in data_wells.iterrows():
        length = len(row["trajectory_x"])
        sl = slice(offset, offset + length)
        data_wells.at[index, "So_current"] = So_current_wells[sl].astype("float32").tolist()
        data_wells.at[index, "eps_so"] = eps_so_wells[sl].astype("float32").tolist()
        data_wells.at[index, "weight_multiplier"] = weight_multipliers[sl].astype("float32").tolist()
        offset += length


def adapt_local_relative_permeability(
    data_wells,
    So_init_wells,
    So_current_wells,
    work_markers,
    well_number,
    weight_multipliers,
    fluid_params,
    relative_permeability,
    reservoir_params,
    options,
):
    """
    Adapt Corey parameters locally so production-well saturation does not exceed its initial value.

    Args:
        data_wells: Well data with trajectory coordinates and production history.
        So_init_wells: Initial oil saturation at trajectory points.
        So_current_wells: Current oil saturation calculated with the base Corey parameters.
        work_markers: Well type for each trajectory point (prod or inj).
        well_number: Well identifier for each trajectory point.
        weight_multipliers: Influence multipliers calculated for inconsistent points.
        fluid_params: Oil and water PVT parameters.
        relative_permeability: Base Corey relative-permeability parameters.
        reservoir_params: Reservoir parameters, including recovery factor.
        options: Local-adaptation limits and eps_so calculation settings.

    Returns:
        tuple:
            np.ndarray: Adapted current oil saturation at trajectory points.
            pd.DataFrame: Well data augmented with point diagnostics and adapted Corey parameters.
    """
    # 1. Приводим входные векторы к единому числовому представлению.
    So_init_wells = np.asarray(So_init_wells, dtype="float64")
    So_current_wells = np.asarray(So_current_wells, dtype="float64")
    work_markers = np.asarray(work_markers)
    well_number = np.asarray(well_number)

    Sor = relative_permeability.Sor
    Swc = relative_permeability.Swc
    Fw = relative_permeability.Fw
    Fo = relative_permeability.Fo
    m1 = relative_permeability.m1
    m2 = relative_permeability.m2

    # 2. Определяем физический нижний предел So и доступную подвижную нефтенасыщенность.
    #    So не должна опускаться ниже Sor и ограничения, заданного коэффициентом извлечения.
    So_limit = np.maximum(Sor, So_init_wells * (1.0 - reservoir_params.KIN))
    mobile_oil = np.maximum(So_init_wells - So_limit, 0.0)

    # 3. Рассчитываем eps_so. При включённой опции допустимое снижение So масштабируется
    #    накопленной добычей скважины и ограничивается доступной подвижной нефтью.
    qo_norm_by_well, qo_base = _production_normalization(data_wells, options.qo_norm_min)
    qo_norm = np.asarray([qo_norm_by_well.get(well, options.qo_norm_min) for well in well_number])

    if options.use_qo_eps_so:
        eps_so = np.maximum(qo_norm * mobile_oil, options.eps_so_base)
    else:
        eps_so = np.full_like(So_init_wells, options.eps_so_base)
    eps_so = np.minimum(eps_so, mobile_oil)

    # 4. Адаптируем ОФП только для конечных точек добывающих скважин с So_current > So_init.
    bad_production = (
        (So_current_wells > So_init_wells)
        & (work_markers == "prod")
        & np.isfinite(So_init_wells)
        & np.isfinite(So_current_wells)
    )

    # Заранее создаём диагностические колонки. Для неадаптированных скважин
    # в них сохраняются исходные параметры Corey.
    for column, value in {
        "OFP_adapted": False,
        "Fw_adapted": Fw,
        "m1_adapted": m1,
        "Fo_adapted": Fo,
        "m2_adapted": m2,
        "Swc_adapted": Swc,
        "Sor_adapted": Sor,
        "eps_so": None,
        "weight_multiplier": None,
    }.items():
        data_wells[column] = value

    if not np.any(bad_production):
        _store_point_arrays(data_wells, So_current_wells, eps_so, weight_multipliers)
        data_wells.attrs["adapted_relative_permeability"] = pd.DataFrame(columns=ADAPTED_OFP_COLUMNS)
        logger.info("Local relative-permeability adaptation was not required")
        return So_current_wells.astype("float32"), data_wells

    # 5. Формируем целевую насыщенность So_target = So_init - eps_so
    #    и последовательно ограничиваем её физически допустимым диапазоном.
    target = np.minimum(So_init_wells - eps_so, So_init_wells)
    target = np.maximum(target, So_limit)
    target = np.minimum(target, 1.0 - Swc - options.eps_so_min)
    target = np.minimum(target, So_init_wells)

    target_local = target[bad_production]
    Sd = np.clip(
        (1.0 - target_local - Swc) / max(1.0 - Sor - Swc, np.finfo("float64").eps),
        1e-9,
        1.0 - 1e-9,
    )
    water_cut_points = np.concatenate(
        [[row.water_cut] * len(row.trajectory_x) for _, row in data_wells.iterrows()]
    ).astype("float64")
    fw = np.clip(water_cut_points[bad_production], 1e-6, 100.0 - 1e-6)

    # 6. По измеренной обводнённости восстанавливаем требуемое отношение krw/kro
    #    в целевой точке насыщенности.
    required_ratio = fw / (100.0 - fw) * fluid_params.mu_w * fluid_params.Bw / (
        fluid_params.mu_o * fluid_params.Bo
    )
    base_ratio = (Fw / Fo) * Sd**m1 / (np.maximum(1.0 - Sd, 1e-12) ** m2)

    # 7. Адаптация выполняется каскадом с заданными физическими границами:
    #    сначала уменьшаем Fw, затем при необходимости увеличиваем m1,
    #    и только после этого уменьшаем m2.
    Fw_multiplier_raw = required_ratio / np.maximum(base_ratio, 1e-30)
    Fw_multiplier = np.clip(Fw_multiplier_raw, options.min_Fw_multiplier, 1.0)
    Fw_local = Fw * Fw_multiplier
    m1_local = np.full_like(Fw_local, m1)
    m2_local = np.full_like(Fw_local, m2)

    need_m1 = Fw_multiplier_raw < options.min_Fw_multiplier
    if np.any(need_m1):
        target_ratio = required_ratio[need_m1] / options.min_Fw_multiplier
        numerator = target_ratio * Fo * (1.0 - Sd[need_m1]) ** m2 / Fw
        m1_required = np.log(np.maximum(numerator, 1e-30)) / np.log(Sd[need_m1])
        m1_local[need_m1] = np.clip(m1_required, m1, m1 * options.max_m1_multiplier)

    ratio_after_Fw_m1 = (Fw_local / Fo) * Sd**m1_local / (np.maximum(1.0 - Sd, 1e-12) ** m2)
    need_m2 = ratio_after_Fw_m1 > required_ratio
    if np.any(need_m2):
        numerator = (Fw_local[need_m2] / Fo) * Sd[need_m2] ** m1_local[need_m2]
        m2_required = np.log(np.maximum(numerator / required_ratio[need_m2], 1e-30)) / np.log(
            1.0 - Sd[need_m2]
        )
        m2_local[need_m2] = np.clip(m2_required, m2 * options.min_m2_multiplier, m2)

    # 8. Параметры сначала вычислены по точкам траектории. Для дальнейшего расчёта
    #    агрегируем их до единого консервативного набора ОФП на скважину.
    point_parameters = pd.DataFrame(
        {
            "well_number": well_number[bad_production],
            "Fw": Fw_local,
            "m1": m1_local,
            "m2": m2_local,
            "adaptation_stage": np.where(need_m2, 3, np.where(need_m1, 2, 1)),
        }
    )
    by_well = point_parameters.groupby("well_number", sort=False).agg(
        {"Fw": "min", "m1": "max", "m2": "min", "adaptation_stage": "max"}
    )

    # 9. Пересчитываем So_current всех точек адаптированной добывающей скважины
    #    уже с её локальным набором параметров Corey.
    adapted = So_current_wells.copy()
    for well, parameters in by_well.iterrows():
        idx = np.where((well_number == well) & (work_markers == "prod"))[0]
        adapted[idx] = [
            1.0
            - get_sw(
                fluid_params.mu_w,
                fluid_params.mu_o,
                fluid_params.Bo,
                fluid_params.Bw,
                water_cut_points[i],
                parameters.Fw,
                parameters.m1,
                Fo,
                parameters.m2,
                Swc,
                Sor,
            )
            for i in idx
        ]

    # 10. Защитный слой: если ограничений параметров ОФП оказалось недостаточно,
    #     принудительно используем ранее рассчитанную целевую насыщенность.
    forced_mask = bad_production & (adapted > So_init_wells)
    adapted[forced_mask] = target[forced_mask]

    # 11. Формируем итоговую таблицу адаптированных ОФП и точечную диагностику,
    #     которые затем доступны пользователю через ResultMaps.
    records = []
    for well, parameters in by_well.iterrows():
        point_mask = well_number == well
        row_mask = data_wells["well_number"] == well
        data_wells.loc[row_mask, "OFP_adapted"] = True
        data_wells.loc[row_mask, "Fw_adapted"] = float(parameters.Fw)
        data_wells.loc[row_mask, "m1_adapted"] = float(parameters.m1)
        data_wells.loc[row_mask, "m2_adapted"] = float(parameters.m2)
        records.append(
            {
                "well_number": well,
                "Fw": float(parameters.Fw),
                "m1": float(parameters.m1),
                "Fo": float(Fo),
                "m2": float(parameters.m2),
                "Swc": float(Swc),
                "Sor": float(Sor),
                "adaptation_stage": {1: "Fw", 2: "Fw_m1", 3: "Fw_m1_m2"}[
                    int(parameters.adaptation_stage)
                ],
                "eps_so_min": float(np.min(eps_so[point_mask])),
                "eps_so_mean": float(np.mean(eps_so[point_mask])),
                "eps_so_max": float(np.max(eps_so[point_mask])),
                "forced_points": int(np.count_nonzero(forced_mask & point_mask)),
            }
        )

    _store_point_arrays(data_wells, adapted, eps_so, weight_multipliers)
    adapted_table = pd.DataFrame.from_records(records, columns=ADAPTED_OFP_COLUMNS)
    data_wells.attrs["adapted_relative_permeability"] = adapted_table

    logger.info(
        "Adapted relative permeability for %d wells and %d trajectory points; safety correction applied to %d points",
        len(adapted_table),
        int(np.count_nonzero(bad_production)),
        int(np.count_nonzero(forced_mask)),
    )
    logger.info(
        "Relative-permeability adaptation stages by well: %s",
        adapted_table["adaptation_stage"].value_counts().to_dict(),
    )
    logger.info(
        "eps_so for adapted points: min=%.6g, mean=%.6g, max=%.6g (Qo normalization base=%s)",
        float(np.min(eps_so[bad_production])),
        float(np.mean(eps_so[bad_production])),
        float(np.max(eps_so[bad_production])),
        "fallback 0.25" if np.isnan(qo_base) else f"{qo_base:.6g}",
    )
    remaining = int(np.count_nonzero(adapted > So_init_wells))
    if remaining:
        logger.warning("Current saturation still exceeds initial saturation at %d well points", remaining)
    return adapted.astype("float32"), data_wells
