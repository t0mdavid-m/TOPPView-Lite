"""The viewer's components, each drawn by its own function.

The Layout page arranges these by name (``CATALOG``) and the viewer draws the
saved arrangement with ``render_layout``. All panels share one
``StateManager``, so a selection in one updates the others wherever they are
placed. A panel draws itself from ``ViewerContext`` and never assumes another
panel was drawn before it.
"""

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import polars as pl
import streamlit as st
from openms_insight import StateManager

from src.components.factory import (
    MIRROR_BOTTOM_IDENTIFIER,
    create_chromatogram_components,
    create_feature_table,
    create_mirror_plot,
)
from src.components.tool_panel import console_panel, operation_panel
from src.export import export_filtered_mzml, filtered_output_path
from src.preprocessing import ensure_chromatogram_input
from src.preprocessing.features import (
    extract_featurexml_to_parquet,
    feature_cache_is_valid,
    find_matching_featurexml,
    get_feature_cache_paths,
)
from src.preprocessing.mirror import ensure_mirror_input

# name -> label, group, and when the component is offered ("always", "ids",
# "im", "features"). The names are stored in saved layouts: do not rename.
CATALOG: List[Dict[str, Any]] = [
    {"name": "chromatogram", "label": "TIC / BPC", "group": "Plots", "needs": "always"},
    {"name": "peak_map", "label": "Peak map", "group": "Plots", "needs": "always"},
    {"name": "spectrum", "label": "Spectrum", "group": "Plots", "needs": "always"},
    {"name": "spectra_table", "label": "Spectra table", "group": "Tables", "needs": "always"},
    {"name": "peak_list", "label": "Peak list", "group": "Tables", "needs": "always"},
    {"name": "im_table", "label": "Ion mobility table", "group": "Tables", "needs": "im"},
    {"name": "id_table", "label": "Identifications", "group": "Identifications", "needs": "ids"},
    {"name": "sequence_view", "label": "Sequence view", "group": "Identifications", "needs": "ids"},
    {"name": "features", "label": "Features table", "group": "Features", "needs": "features"},
    {"name": "tools", "label": "Tools", "group": "Tools", "needs": "always"},
    {"name": "export", "label": "Export filtered mzML", "group": "Tools", "needs": "always"},
]

DEFAULT_LAYOUT: List[List[str]] = [
    ["tools"],
    ["im_table"],
    ["chromatogram"],
    ["peak_map"],
    ["spectra_table", "peak_list"],
    ["spectrum"],
    ["sequence_view"],
    ["id_table"],
    ["features"],
    ["export"],
]

_NEEDS_NOTE = {
    "ids": "when an idXML matches the mzML",
    "im": "for ion mobility data",
    "features": "when a featureXML matches the mzML",
}


def editor_catalog() -> List[Dict[str, Any]]:
    """Catalog entries for the layout editor, with when each one is shown."""
    out = []
    for entry in CATALOG:
        item = {k: entry[k] for k in ("name", "label", "group")}
        note = _NEEDS_NOTE.get(entry["needs"])
        if note:
            item["label"] = f"{entry['label']} ({note})"
        out.append(item)
    return out


def layout_path(workspace: Path) -> Path:
    return Path(workspace, "layouts", "viewer.json")


@dataclass
class ViewerContext:
    """Everything the panels need about the file on screen."""

    workspace: Path
    selected_file: Path
    paths: Dict[str, Path]
    im_info: Dict[str, Any]
    file_id: str
    cache_path: str
    state_manager: StateManager
    ms_components: Dict[str, Any]
    id_components: Optional[Dict[str, Any]] = None
    id_paths: Optional[Dict[str, Path]] = None
    search_params: Optional[Dict[str, Any]] = None
    featurexml_path: Optional[Path] = None
    local: bool = False

    @property
    def file_key(self) -> str:
        return self.selected_file.stem

    @property
    def has_ids(self) -> bool:
        return self.id_components is not None

    @property
    def has_im(self) -> bool:
        return self.ms_components.get("im_table") is not None


def available_catalog(ctx: ViewerContext) -> List[Dict[str, Any]]:
    """The catalog entries this file can show."""
    have = {
        "always": True,
        "ids": ctx.has_ids,
        "im": ctx.has_im,
        "features": ctx.featurexml_path is not None,
    }
    return [e for e in CATALOG if have[e["needs"]]]


# =============================================================================
# Panels
# =============================================================================


def chromatogram(ctx: ViewerContext) -> None:
    """TIC or BPC of the MS1 spectra. A click selects that MS1 spectrum."""
    ensure_chromatogram_input(ctx.paths, ctx.im_info)
    plots = create_chromatogram_components(ctx.paths, ctx.im_info, ctx.file_id)
    kind = st.segmented_control(
        "Chromatogram",
        options=["tic", "bpc"],
        format_func={"tic": "TIC", "bpc": "BPC"}.get,
        default="tic",
        key="viewer_chromatogram_kind",
        label_visibility="collapsed",
    ) or "tic"
    plots[kind](key=f"{ctx.file_key}_{kind}_plot", state_manager=ctx.state_manager, height=250)


def _feature_paths(ctx: ViewerContext, consume: bool = False) -> tuple[dict, bool]:
    """Parquet cache of the matching featureXML, (re)built when stale. Returns
    the paths and whether the cache was rebuilt; ``consume`` (the features
    table) also reports a rebuild made earlier by the peak map, once."""
    feature_paths = get_feature_cache_paths(ctx.workspace, ctx.featurexml_path)
    rebuilt = not feature_cache_is_valid(ctx.featurexml_path, feature_paths)
    stale_key = f"viewer_feature_table_stale_{ctx.file_key}"
    if rebuilt:
        with st.spinner(f"Reading {ctx.featurexml_path.name}..."):
            extract_featurexml_to_parquet(
                ctx.featurexml_path, feature_paths, ctx.paths["metadata"]
            )
        # The features table rebuilds its cache too, also when the peak map
        # came first and found the parquet stale
        st.session_state[stale_key] = True
    return feature_paths, st.session_state.pop(stale_key, False) if consume else rebuilt


def _show_selected_feature(ctx: ViewerContext, feature_paths: dict) -> None:
    """A newly selected feature (in the table or on the peak map) shows the MS1
    spectrum at its apex."""
    state_manager = ctx.state_manager
    selected_feature = state_manager.get_selection("feature")
    applied_key = f"viewer_applied_feature_{ctx.file_key}"
    if selected_feature is None or st.session_state.get(applied_key) == selected_feature:
        return
    st.session_state[applied_key] = selected_feature
    apex = (
        pl.scan_parquet(feature_paths["features"])
        .filter(pl.col("feature_id") == selected_feature)
        .select("apex_scan_id")
        .collect()
    )
    if apex.height and apex["apex_scan_id"][0] > 0:
        state_manager.set_selection("spectrum", int(apex["apex_scan_id"][0]))
        state_manager.clear_selection("peak")
        state_manager.clear_selection("identification")
        st.rerun()


def peak_map(ctx: ViewerContext) -> None:
    """MS1 peak map. With a matching featureXML, its features are drawn as boxes
    (RT and m/z extent of their mass traces); clicking inside one selects it."""
    heatmap = ctx.ms_components["heatmap"]
    # Heatmap.set_boxes() arrived after openms-insight 0.4.0
    can_draw = hasattr(heatmap, "set_boxes")
    boxes = None
    if can_draw and ctx.featurexml_path is not None and st.toggle(
        "Show features", value=True, key="viewer_peak_map_features"
    ):
        feature_paths, _ = _feature_paths(ctx)
        # Most intense first: they are the ones kept when a view has too many
        boxes = (
            pl.scan_parquet(feature_paths["features"])
            .select("feature_id", "intensity", "rt_start", "rt_end", "mz_start", "mz_end")
            .sort("intensity", descending=True)
            .collect()
        )
        _show_selected_feature(ctx, feature_paths)
    if can_draw:
        heatmap.set_boxes(
            boxes,
            x_range=("rt_start", "rt_end"),
            y_range=("mz_start", "mz_end"),
            id_column="feature_id",
            selection="feature",
            label="Features",
        )
    heatmap(key=f"{ctx.file_key}_heatmap", state_manager=ctx.state_manager, height=400)


def spectra_table(ctx: ViewerContext) -> None:
    # With identifications the table has a # IDs column
    table = (
        ctx.id_components["spectra_table_with_ids"]
        if ctx.has_ids
        else ctx.ms_components["spectra_table"]
    )
    table(key=f"{ctx.file_key}_spectra_table", state_manager=ctx.state_manager, height=400)


def peak_list(ctx: ViewerContext) -> None:
    ctx.ms_components["peaks_table"](
        key=f"{ctx.file_key}_peaks_table", state_manager=ctx.state_manager, height=400
    )


def im_table(ctx: ViewerContext) -> None:
    ctx.ms_components["im_table"](
        key=f"{ctx.file_key}_im_table", state_manager=ctx.state_manager, height=400
    )


def spectrum(ctx: ViewerContext) -> None:
    """The selected spectrum: plain, annotated with the selected identification,
    or mirrored against its theoretical b/y ions."""
    if not ctx.has_ids:
        ctx.ms_components["spectrum_plot"](
            key=f"{ctx.file_key}_spectrum_plot", state_manager=ctx.state_manager, height=400
        )
        return

    mode = st.segmented_control(
        "Spectrum view",
        ["Annotated", "Mirror vs. theoretical"],
        default="Annotated",
        key="viewer_spectrum_mode",
        label_visibility="collapsed",
    )
    if mode != "Mirror vs. theoretical":
        ctx.id_components["annotated_spectrum_plot"](
            key=f"{ctx.file_key}_spectrum_plot",
            state_manager=ctx.state_manager,
            height=400,
            sequence_view_key=f"{ctx.file_key}_sequence_view",
        )
        return

    state_manager = ctx.state_manager
    mirror_key = f"viewer_mirror_{ctx.file_key}"
    mirror_paths, rebuilt = ensure_mirror_input(ctx.paths, ctx.id_paths, ctx.search_params)
    if rebuilt or mirror_key not in st.session_state:
        with st.spinner("Generating theoretical spectra..."):
            st.session_state[mirror_key] = create_mirror_plot(
                mirror_paths["mirror"], ctx.cache_path, ctx.file_id, regenerate_cache=rebuilt
            )

    # The bottom half follows the selected identification.
    selected_id = state_manager.get_selection("identification")
    if state_manager.get_selection(MIRROR_BOTTOM_IDENTIFIER) != selected_id:
        if selected_id is None:
            state_manager.clear_selection(MIRROR_BOTTOM_IDENTIFIER)
        else:
            state_manager.set_selection(MIRROR_BOTTOM_IDENTIFIER, selected_id)

    if selected_id is None:
        st.caption("Select an identification to compare its spectrum with the theoretical b/y ions.")
    else:
        match = (
            pl.scan_parquet(mirror_paths["summary"])
            .filter(pl.col("id_idx") == selected_id)
            .collect()
        )
        if match.height:
            row = match.row(0, named=True)
            st.caption(
                f"{row['matched_ions']} of {row['total_ions']} theoretical b/y ions matched "
                f"({row['coverage']:.0f} %). Matched peaks are red and labelled."
            )
        else:
            st.caption("This identification is not linked to a spectrum.")

    st.session_state[mirror_key](
        key=f"{ctx.file_key}_mirror_plot", state_manager=state_manager, height=450
    )


def sequence_view(ctx: ViewerContext) -> None:
    ctx.id_components["sequence_view"](
        key=f"{ctx.file_key}_sequence_view", state_manager=ctx.state_manager, height=800
    )


def id_table(ctx: ViewerContext) -> None:
    ctx.id_components["id_table"](
        key=f"{ctx.file_key}_id_table", state_manager=ctx.state_manager, height=300
    )


def features(ctx: ViewerContext) -> None:
    """Features of the featureXML named like the mzML. A click shows the MS1
    spectrum at the feature's apex."""
    featurexml_path = ctx.featurexml_path
    state_manager = ctx.state_manager
    feature_paths, rebuilt = _feature_paths(ctx, consume=True)
    table = create_feature_table(
        str(feature_paths["features"]), ctx.cache_path, ctx.file_id, regenerate_cache=rebuilt
    )
    st.caption(
        f"Features from {featurexml_path.name}. Click a feature to show the MS1 "
        "spectrum at its apex."
    )
    table(key=f"{ctx.file_key}_feature_table", state_manager=state_manager, height=350)
    _show_selected_feature(ctx, feature_paths)


def tools(ctx: ViewerContext) -> None:
    """Operations (and, locally, the Python console) on this file. A featureXML
    named like the mzML fills the features table right away; a new mzML is opened."""
    with st.expander("Tools", icon="🧰"):
        tabs = st.tabs(["Operations"] + (["Python console"] if ctx.local else []))
        with tabs[0]:
            written = operation_panel(ctx.workspace, ctx.selected_file, key="viewer_tools")
        if len(tabs) > 1:
            with tabs[1]:
                written += console_panel(
                    ctx.workspace, ctx.selected_file, key="viewer_console",
                    scan_id=ctx.state_manager.get_selection("spectrum"),
                )
    new_mzml = [p for p in written if p.suffix == ".mzML" and p.name != ctx.selected_file.name]
    if new_mzml:
        st.session_state["viewer_pending_file"] = new_mzml[-1].name
        st.rerun()
    # A new or replaced featureXML for this file: rerun so the features panel
    # shows it wherever it is placed, also above this one.
    if any(
        p.suffix.lower() == ".featurexml" and p.stem.lower() == ctx.selected_file.stem.lower()
        for p in written
    ):
        st.rerun()


@st.fragment
def export(ctx: ViewerContext) -> None:
    """Write a copy of the mzML restricted to an RT range, m/z range and MS levels."""
    file_key = ctx.file_key
    with st.expander("Export filtered mzML"):
        meta = pl.read_parquet(ctx.paths["metadata"], columns=["retention_time", "ms_level"])
        mz_bounds = (
            pl.scan_parquet(ctx.paths["peaks"])
            .select(pl.col("mass").min().alias("lo"), pl.col("mass").max().alias("hi"))
            .collect()
            .row(0)
        )
        rt_lo = float(np.floor(meta["retention_time"].min() * 100) / 100)
        rt_hi = float(np.ceil(meta["retention_time"].max() * 100) / 100)
        mz_lo = float(np.floor(mz_bounds[0] or 0.0))
        mz_hi = float(np.ceil(mz_bounds[1] or 0.0))
        levels = sorted(int(v) for v in meta["ms_level"].unique().to_list())

        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            rt_range = st.slider(
                "RT range (min)", rt_lo, max(rt_hi, rt_lo + 0.01), (rt_lo, max(rt_hi, rt_lo + 0.01)),
                step=0.01, key=f"export_rt_{file_key}",
            )
        with c2:
            mz_range = st.slider(
                "m/z range", mz_lo, max(mz_hi, mz_lo + 1.0), (mz_lo, max(mz_hi, mz_lo + 1.0)),
                step=1.0, key=f"export_mz_{file_key}",
            )
        with c3:
            ms_levels = st.multiselect(
                "MS levels", levels, default=levels, key=f"export_levels_{file_key}"
            )

        out_path = filtered_output_path(ctx.workspace, ctx.selected_file)
        result_key = f"export_result_{file_key}"
        if st.button("Create filtered mzML", disabled=not ms_levels):
            with st.spinner("Filtering..."):
                try:
                    backend = export_filtered_mzml(
                        ctx.selected_file,
                        out_path,
                        # Widgets show minutes; mzML stores seconds. Widen by a hair so
                        # spectra on the rounded bounds are kept.
                        (rt_range[0] * 60 - 0.01, rt_range[1] * 60 + 0.01),
                        mz_range,
                        ms_levels,
                    )
                    st.session_state[result_key] = {"ok": True, "backend": backend}
                except (ValueError, RuntimeError) as err:
                    st.session_state[result_key] = {"ok": False, "error": str(err)}

        result = st.session_state.get(result_key)
        if result and not result["ok"]:
            st.error(result["error"])
        elif result and out_path.exists():
            size_mb = out_path.stat().st_size / 1e6
            st.caption(f"{out_path.name}: {size_mb:.1f} MB, written with {result['backend']}.")
            with open(out_path, "rb") as fh:
                st.download_button(
                    "Download filtered mzML", fh, file_name=out_path.name, mime="application/xml"
                )


PANELS: Dict[str, Callable[[ViewerContext], None]] = {
    "chromatogram": chromatogram,
    "peak_map": peak_map,
    "spectrum": spectrum,
    "spectra_table": spectra_table,
    "peak_list": peak_list,
    "im_table": im_table,
    "id_table": id_table,
    "sequence_view": sequence_view,
    "features": features,
    "tools": tools,
    "export": export,
}


def renderers(ctx: ViewerContext) -> Dict[str, Callable[[], None]]:
    """Renderers for ``render_layout``, bound to the file on screen."""
    return {name: partial(fn, ctx) for name, fn in PANELS.items()}

