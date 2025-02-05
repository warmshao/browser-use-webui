Feature: Visual Recognition
    As a user
    I want to verify visual elements on websites
    So that I can ensure proper content display

    Scenario: Verify company logos
        Given I am on "apple.com"
        Then I should see the company logo for Apple
        When I navigate to "microsoft.com"
        Then I should see the company logo for Microsoft
        When I navigate to "google.com"
        Then I should see the company logo for Google

    Scenario: Check image content
        Given I am on the Google homepage
        When I search for "Eiffel Tower" with vision enabled
        Then I should see search results with images
        And I should see images of the Eiffel Tower
        When I search for "Golden Gate Bridge" with vision enabled
        Then I should see images of the Golden Gate Bridge

    Scenario: Verify page layout
        Given I am on "github.com"
        Then I should see a page element: navigation bar at the top
        And I should see a page element: search bar
        When I switch to dark mode
        Then the background should be dark
        And text should be light colored 