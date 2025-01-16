import gradio as gr
from plugins.plugin_base import PluginBase
import logging
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class MockPlugin(PluginBase):
    def __init__(self, manifest: Optional[Dict[str, Any]] = None):
        # Call parent class initialization first
        super().__init__(manifest=manifest)
        
        # Initialize plugin state
        self.set_state('last_init', datetime.now().isoformat())
        logger.info(f"{self.name} initialized")

    def is_enabled(self) -> bool:
        has_permissions, _ = self._validate_permissions()
        return self.enabled and has_permissions

    def on_enable(self) -> None:
        """Handle plugin enabling."""
        self.enabled = True
        self.set_state('last_enabled', datetime.now().isoformat())
        logger.info(f"{self.name} enabled")

    def on_disable(self) -> None:
        """Handle plugin disabling."""
        self.enabled = False
        self.set_state('last_disabled', datetime.now().isoformat())
        logger.info(f"{self.name} disabled")

    def on_unload(self) -> None:
        """Clean up before unloading."""
        self.set_state('last_unloaded', datetime.now().isoformat())
        super().on_unload()  # Don't forget to call parent's on_unload
        logger.info(f"{self.name} unloaded")

    def create_ui(self, main_tabs: gr.Tabs) -> None:
        """Create a simple UI for demonstration."""
        try:
            with main_tabs:
                with gr.TabItem("Mock Plugin"):
                    gr.Markdown(f"## {self.name} v{self.version}")
                    gr.Markdown(f"_{self.description}_")
                    
                    with gr.Row():
                        text_input = gr.Textbox(
                            label="Input Text",
                            placeholder="Enter some text"
                        )
                        output_text = gr.Textbox(
                            label="Output",
                            interactive=False
                        )
                    
                    def process_text(text: str) -> str:
                        # Sanitize input
                        text = self._sanitize_input(text)
                        # Store last processed text in state
                        self.set_state('last_processed', text)
                        return f"Processed: {text.upper()}"
                    
                    process_button = gr.Button("Process")
                    process_button.click(
                        fn=process_text,
                        inputs=[text_input],
                        outputs=[output_text]
                    )
                    
                    # Plugin state display
                    with gr.Row():
                        gr.Markdown("### Plugin State")
                        state_text = gr.Markdown(
                            f"""
                            - Last Initialized: {self.get_state('last_init', 'Never')}
                            - Last Enabled: {self.get_state('last_enabled', 'Never')}
                            - Last Disabled: {self.get_state('last_disabled', 'Never')}
                            - Last Processed Text: {self.get_state('last_processed', 'None')}
                            """
                        )
                    
        except Exception as e:
            logger.error(f"Failed to create Mock plugin UI: {str(e)}")
            raise 