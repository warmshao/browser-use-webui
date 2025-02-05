Feature: Browser Search Functionality
    As a user
    I want to perform searches in the browser
    So that I can find relevant information

    Scenario: Search on Google
        Given I am on the Google homepage
        When I search for "OpenAI"
        Then I should see search results
        And the first result should contain "openai.com"

    Scenario: Search with Images
        Given I am on the Google homepage
        When I search for "Nvidia" with vision enabled
        Then I should see search results with images
        And I should see the Nvidia logo 