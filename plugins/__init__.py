import os
import yaml
import logging
from typing import List, Dict, Any, Optional, Tuple
from .plugin_base import PluginBase
from .factory import PluginFactory

logger = logging.getLogger(__name__)

class PluginManager:
    """Manages plugin lifecycle and state."""
    
    def __init__(self):
        self.loaded_plugins: Dict[str, PluginBase] = {}
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load the centralized plugin configuration."""
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load plugin config: {str(e)}")
            return {
                "enabled_plugins": [],
                "plugin_config": {},
                "security": {"sanitize_all_inputs": True}
            }
    
    def save_config(self) -> None:
        """Save the current configuration."""
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        try:
            with open(config_path, 'w') as f:
                yaml.safe_dump(self.config, f)
        except Exception as e:
            logger.error(f"Failed to save plugin config: {str(e)}")
    
    def enable_plugin(self, plugin_name: str) -> Tuple[bool, str]:
        """Enable a plugin."""
        if plugin_name not in self.loaded_plugins:
            return False, f"Plugin {plugin_name} not found"
        
        plugin = self.loaded_plugins[plugin_name]
        if plugin_name not in self.config["enabled_plugins"]:
            self.config["enabled_plugins"].append(plugin_name)
            self.save_config()
            plugin.on_enable()
        return True, f"Plugin {plugin_name} enabled"
    
    def disable_plugin(self, plugin_name: str) -> Tuple[bool, str]:
        """Disable a plugin."""
        if plugin_name not in self.loaded_plugins:
            return False, f"Plugin {plugin_name} not found"
        
        if plugin_name in self.config["enabled_plugins"]:
            self.config["enabled_plugins"].remove(plugin_name)
            self.save_config()
            self.loaded_plugins[plugin_name].on_disable()
        return True, f"Plugin {plugin_name} disabled"
    
    def unload_plugin(self, plugin_name: str) -> Tuple[bool, str]:
        """Unload a plugin."""
        if plugin_name not in self.loaded_plugins:
            return False, f"Plugin {plugin_name} not found"
        
        plugin = self.loaded_plugins[plugin_name]
        plugin.on_unload()
        del self.loaded_plugins[plugin_name]
        return True, f"Plugin {plugin_name} unloaded"
    
    def load_plugins(self) -> List[PluginBase]:
        """Load plugins based on the centralized configuration."""
        enabled_plugins = self.config.get("enabled_plugins", [])
        plugins_dir = os.path.dirname(__file__)
        webui_version = self.config.get("webui_version", "1.0.0")
        
        logger.info(f"Loading enabled plugins: {enabled_plugins}")
        
        # Clear existing plugins
        self.loaded_plugins.clear()
        
        # Load enabled plugins using factory
        for plugin_name in enabled_plugins:
            # Create plugin using factory
            plugin = PluginFactory.create_plugin(plugins_dir, plugin_name)
            if not plugin:
                continue
            
            # Check compatibility
            is_compatible, reason = plugin.validate_compatibility(webui_version)
            if not is_compatible:
                logger.warning(
                    f"Plugin {plugin.name} v{plugin.version} is not compatible: {reason}"
                )
                continue
            
            # Check permissions
            has_permissions, reason = plugin._validate_permissions()
            if not has_permissions:
                logger.warning(f"Plugin {plugin.name} permission denied: {reason}")
                continue
            
            # Apply config overrides
            plugin_config = self.config.get("plugin_config", {}).get(plugin_name, {})
            for key, value in plugin_config.items():
                plugin.set_config(key, value)
            
            # Store plugin
            self.loaded_plugins[plugin_name] = plugin
            logger.info(f"Successfully loaded plugin: {plugin.name} v{plugin.version}")
        
        logger.info(f"Successfully loaded {len(self.loaded_plugins)} plugins")
        return list(self.loaded_plugins.values())

# Global plugin manager instance
plugin_manager = PluginManager()

def load_plugins() -> List[PluginBase]:
    """Convenience function to load plugins using the global manager."""
    return plugin_manager.load_plugins() 