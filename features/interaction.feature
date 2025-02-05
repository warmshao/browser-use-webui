Feature: Browser Interaction
    As a user
    I want to interact with web elements
    So that I can perform actions on websites

    Scenario: Fill and submit a form
        Given I am on the Google homepage
        When I type "Python programming" in the search box
        And I click the search button
        Then I should see search results
        And the page title should contain "Python programming"

    Scenario: Click and verify links
        Given I am on the Google homepage
        When I search for "OpenAI"
        And I click the first link
        Then I should be on "openai.com"
        And I should see the OpenAI logo

    Scenario: Scroll and interact
        Given I am on "github.com"
        When I scroll to the bottom of the page
        Then I should see the footer
        When I scroll to the top
        Then I should see the header 