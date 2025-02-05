Feature: Browser Navigation
    As a user
    I want to navigate between different websites
    So that I can access various web content

    Scenario: Navigate to multiple sites
        Given I am on the Google homepage
        When I navigate to "github.com"
        Then I should see "GitHub" in the page
        When I navigate to "openai.com"
        Then I should see "OpenAI" in the page

    Scenario: Use browser history
        Given I am on the Google homepage
        When I navigate to "github.com"
        And I go back in browser history
        Then I should be on "google.com"
        When I go forward in browser history
        Then I should be on "github.com" 