import gradio as gr
import os
from plugins.plugin_base import PluginBase
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class TwitterPlugin(PluginBase):
    def __init__(self, manifest: Optional[Dict[str, Any]] = None):
        # Call parent class initialization first
        super().__init__(manifest=manifest)
        
        # Initialize mock state
        self.tweets = []
        self.logged_in = False
        
    def is_enabled(self) -> bool:
        has_permissions, _ = self._validate_permissions()
        return self.enabled and has_permissions

    def create_ui(self, main_tabs: gr.Tabs) -> None:
        """Attach the plugin's UI elements to the main web UI."""
        try:
            with main_tabs:
                with gr.TabItem("Mock Twitter"):
                    gr.Markdown(f"## {self.name} v{self.version}")
                    gr.Markdown(f"_{self.description}_")
                    
                    with gr.Row():
                        username = gr.Textbox(
                            label="Twitter Username",
                            placeholder="Enter your Twitter username"
                        )
                        password = gr.Textbox(
                            label="Password",
                            type="password",
                            placeholder="Enter your password"
                        )
                    
                    def mock_login(username: str, password: str) -> str:
                        if not username or not password:
                            return "Please enter both username and password"
                        self.logged_in = True
                        return f"Logged in as {username} (Mock)"
                    
                    login_button = gr.Button("Login")
                    login_result = gr.Textbox(label="Login Status")
                    login_button.click(
                        fn=mock_login,
                        inputs=[username, password],
                        outputs=[login_result]
                    )
                    
                    def mock_post_tweet(tweet_content: str) -> str:
                        if not self.logged_in:
                            return "Please log in first"
                            
                        # Sanitize input
                        tweet_content = self._sanitize_input(tweet_content)
                        if not tweet_content:
                            return "Tweet content cannot be empty"
                            
                        # Store tweet in mock state
                        self.tweets.append(tweet_content)
                        return f"Tweet posted successfully! (Mock) Total tweets: {len(self.tweets)}"
                    
                    with gr.Row():
                        tweet_text = gr.Textbox(
                            label="Tweet Content",
                            lines=3,
                            placeholder="What's happening?"
                        )
                        post_button = gr.Button("Post Tweet")
                    post_result = gr.Textbox(label="Post Result")
                    
                    def mock_search_tweets(query: str) -> str:
                        if not self.logged_in:
                            return "Please log in first"
                            
                        # Sanitize input
                        query = self._sanitize_input(query)
                        if not query:
                            return "Search query cannot be empty"
                            
                        # Search in mock tweets
                        matching_tweets = [
                            tweet for tweet in self.tweets 
                            if query.lower() in tweet.lower()
                        ]
                        
                        if not matching_tweets:
                            return "No tweets found matching your query"
                            
                        return "\n\n".join([
                            f"Tweet {i+1}: {tweet}" 
                            for i, tweet in enumerate(matching_tweets)
                        ])
                    
                    with gr.Row():
                        search_query = gr.Textbox(
                            label="Search Query",
                            placeholder="Enter search terms"
                        )
                        search_button = gr.Button("Search Tweets")
                    search_result = gr.Textbox(label="Search Results", lines=10)
                    
                    # Connect the buttons to their functions
                    post_button.click(
                        fn=mock_post_tweet,
                        inputs=[tweet_text],
                        outputs=[post_result]
                    )
                    
                    search_button.click(
                        fn=mock_search_tweets,
                        inputs=[search_query],
                        outputs=[search_result]
                    )
                    
        except Exception as e:
            logger.error(f"Failed to create Twitter plugin UI: {str(e)}")
            raise 