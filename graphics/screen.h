#pragma once

// The app's screens. main.cpp draws the current one each frame; a screen's
// render function returns which screen to show next.
enum class Screen {
    Title,               // main_screen.cpp: title and Begin button
    Menu,                // menu_screen.cpp: View Research or Classifier
    Research,            // research_screen.cpp: paper download or Under the Hood
    UnderTheHood,        // under_the_hood_screen.cpp: the explainer slides
    Demo,                // demo_screen.cpp: the Classifier on real LIGO data
    SpacetimeSimulation, // spacetime_simulation_screen.cpp: Earth vs. black hole curvature
    Simulation,          // simulation_screen.cpp: 3D replay of one merger
};
