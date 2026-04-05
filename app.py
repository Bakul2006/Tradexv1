"""Root entrypoint for local OpenEnv runs and Hugging Face Spaces."""

from __future__ import annotations

import os

import gradio as gr

from meverse.server.app import app as openenv_app
from meverse.server.app import main as openenv_main
from meverse.models import SurveillanceAction, SurveillanceObservation
from meverse.server.meverse_environment import MarketSurveillanceEnvironment

SPACE_THEME = None
SPACE_CSS = None


def _running_in_hf_space() -> bool:
    return any(os.getenv(name) for name in ("SPACE_ID", "SPACE_AUTHOR_NAME", "HF_SPACE_ID"))


def _app_mode() -> str:
    if _running_in_hf_space():
        return "space"
    return os.getenv("TRADEX_APP_MODE", "openenv").strip().lower()


def _build_space_app() -> gr.Blocks:
    global SPACE_THEME, SPACE_CSS

    from dashboard import CUSTOM_CSS as DASHBOARD_CSS
    from dashboard import THEME as DASHBOARD_THEME
    from dashboard import build_app as build_dashboard_app

    # Try to import OpenEnv web interface for the Playground tab.
    # If unavailable (version mismatch, missing deps), fall back to dashboard-only.
    playground_blocks = None
    openenv_css = ""
    openenv_theme = None
    try:
        from openenv.core.env_server.web_interface import (
            OPENENV_GRADIO_CSS,
            OPENENV_GRADIO_THEME,
            WebInterfaceManager,
            _extract_action_fields,
            _is_chat_env,
            build_gradio_app,
            get_gradio_display_title,
            get_quick_start_markdown,
            load_environment_metadata,
        )

        env_name = "amm-market-surveillance"
        metadata = load_environment_metadata(MarketSurveillanceEnvironment, env_name)
        web_manager = WebInterfaceManager(
            MarketSurveillanceEnvironment,
            SurveillanceAction,
            SurveillanceObservation,
            metadata,
        )
        action_fields = _extract_action_fields(SurveillanceAction)
        is_chat_env = _is_chat_env(SurveillanceAction)
        quick_start_md = get_quick_start_markdown(
            metadata,
            SurveillanceAction,
            SurveillanceObservation,
        )
        title = get_gradio_display_title(metadata, fallback="TradeX")
        playground_blocks = build_gradio_app(
            web_manager,
            action_fields,
            metadata,
            is_chat_env,
            title=title,
            quick_start_md=quick_start_md,
        )
        openenv_css = OPENENV_GRADIO_CSS
        openenv_theme = OPENENV_GRADIO_THEME
    except Exception:
        pass

    dashboard_blocks = build_dashboard_app()

    SPACE_CSS = "\n".join(
        filter(None, [
            openenv_css,
            """
            .space-shell { padding: 0 !important; }
            .space-tabs > .tab-nav {
                background: transparent !important;
                border-bottom: 1px solid rgba(139, 148, 158, 0.25) !important;
                margin-bottom: 8px !important;
            }
            .space-tabs > .tab-nav button {
                font-weight: 600 !important;
            }
            """,
            DASHBOARD_CSS,
        ])
    )
    SPACE_THEME = openenv_theme or DASHBOARD_THEME

    title = "TradeX Surveillance Dashboard"
    with gr.Blocks(
        title=title,
        fill_width=True,
        elem_classes=["space-shell"],
    ) as blocks_app:
        if playground_blocks is not None:
            with gr.Tabs(elem_classes=["space-tabs"]):
                with gr.Tab("Playground"):
                    playground_blocks.render()
                with gr.Tab("Dashboard"):
                    dashboard_blocks.render()
        else:
            # Playground unavailable — show dashboard directly
            dashboard_blocks.render()

    return blocks_app


def _make_asgi_app():
    """Wrap the Gradio Blocks into a launchable ASGI app for uvicorn."""
    blocks = _build_space_app()
    # Gradio's launch() returns (app, local_url, share_url).
    # Calling with prevent_thread_lock=True starts the server internally
    # but we need the underlying FastAPI app for uvicorn.
    # Use the blocks' internal FastAPI app instead.
    gradio_app = gr.routes.App.create_app(blocks)
    return gradio_app


if _app_mode() == "space":
    # For HF Spaces / uvicorn: expose a proper ASGI app
    app = _make_asgi_app()

    def main() -> None:
        port = int(os.getenv("PORT", "7860"))
        # When run directly (python app.py), launch via Gradio
        space_blocks = _build_space_app()
        space_blocks.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            theme=SPACE_THEME,
            css=SPACE_CSS,
        )
else:
    app = openenv_app
    main = openenv_main


__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
