#pragma once

#include "imgui.h"
#include "../models/state.h" // For ThemeProfile enum

namespace afs::viz::themes {

/// Apply the AFS dark theme to ImGui with specific profiles.
inline void ApplyHafsTheme(ThemeProfile profile = ThemeProfile::Cobalt) {
  ImGuiStyle& style = ImGui::GetStyle();

  // Rounding & Padding for a GIMP-like technical look
  style.WindowRounding = 0.0f;  // Professional tools often use square windows
  style.FrameRounding = 2.0f;
  style.GrabRounding = 2.0f;
  style.PopupRounding = 2.0f;
  style.ScrollbarRounding = 2.0f;
  style.TabRounding = 0.0f;     // Square tabs
  
  style.WindowPadding = ImVec2(8, 8);   // Tighter
  style.FramePadding = ImVec2(6, 4);    // Compact input fields
  style.ItemSpacing = ImVec2(8, 4);     // Denser layout
  style.ItemInnerSpacing = ImVec2(4, 4);
  style.CellPadding = ImVec2(4, 2);
  style.ScrollbarSize = 14.0f;
  style.GrabMinSize = 10.0f;

  ImVec4* colors = style.Colors;

  // Base background (Deep neutral)
  colors[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.06f, 0.08f, 1.00f);
  colors[ImGuiCol_ChildBg] = ImVec4(0.06f, 0.06f, 0.08f, 0.90f); // Darker, less transparent
  colors[ImGuiCol_PopupBg] = ImVec4(0.08f, 0.08f, 0.10f, 0.98f);
  colors[ImGuiCol_Border] = ImVec4(1.0f, 1.0f, 1.0f, 0.08f);
  
  // Profile specific colors
  ImVec4 primary, secondary, accent;
  
  if (profile == ThemeProfile::Cobalt) {
      primary = ImVec4(0.0f, 0.48f, 1.0f, 1.00f);     // Vivid Azure
      secondary = ImVec4(0.12f, 0.14f, 0.18f, 1.00f); // Darker Midnight for contrast
      accent = ImVec4(0.0f, 0.85f, 1.0f, 1.0f);       // Neon Cyan
  } else if (profile == ThemeProfile::Amber) {
      primary = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);   // Bright Orange
      secondary = ImVec4(0.20f, 0.10f, 0.05f, 1.00f); // Deep Rust
      accent = ImVec4(1.00f, 0.90f, 0.30f, 1.00f);    // Electric Gold
  } else if (profile == ThemeProfile::Emerald) {
      primary = ImVec4(0.00f, 0.85f, 0.45f, 1.00f);   // Neon Green
      secondary = ImVec4(0.05f, 0.15f, 0.10f, 1.00f); // Dark Jungle
      accent = ImVec4(0.40f, 1.00f, 0.60f, 1.00f);    // Bright Mint
  } else if (profile == ThemeProfile::Cyberpunk) {
      primary = ImVec4(1.0f, 0.0f, 0.5f, 1.0f);       // Hot Pink
      secondary = ImVec4(0.1f, 0.0f, 0.2f, 1.0f);     // Deep Purple
      accent = ImVec4(0.0f, 1.0f, 1.0f, 1.0f);       // Cyan
  } else if (profile == ThemeProfile::Monochrome) {
      primary = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
      secondary = ImVec4(0.1f, 0.1f, 0.1f, 1.00f);
      accent = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
  } else if (profile == ThemeProfile::Solarized) {
      primary = ImVec4(0.15f, 0.45f, 0.55f, 1.0f);    // Blue
      secondary = ImVec4(0.03f, 0.21f, 0.26f, 1.00f); // Base03
      accent = ImVec4(0.52f, 0.60f, 0.00f, 1.0f);     // Green
  } else if (profile == ThemeProfile::Nord) {
      primary = ImVec4(0.53f, 0.75f, 0.82f, 1.0f);    // Frost
      secondary = ImVec4(0.18f, 0.20f, 0.25f, 1.00f); // Polar Night
      accent = ImVec4(0.56f, 0.80f, 0.71f, 1.0f);     // Frost/Teal
  } else if (profile == ThemeProfile::Dracula) {
      primary = ImVec4(0.74f, 0.57f, 0.97f, 1.0f);    // Purple
      secondary = ImVec4(0.16f, 0.17f, 0.24f, 1.00f); // Background
      accent = ImVec4(1.00f, 0.47f, 0.77f, 1.0f);     // Pink
  } else { // Emerald (Fallback if Default or Cobalt)
      primary = ImVec4(0.00f, 0.85f, 0.45f, 1.00f);   // Neon Green
      secondary = ImVec4(0.05f, 0.15f, 0.10f, 1.00f); // Dark Jungle
      accent = ImVec4(0.40f, 1.00f, 0.60f, 1.00f);    // Bright Mint
  }

  // Apply profile to components
  colors[ImGuiCol_Header] = secondary;
  colors[ImGuiCol_HeaderHovered] = primary;
  colors[ImGuiCol_HeaderActive] = accent;

  colors[ImGuiCol_Button] = secondary;
  colors[ImGuiCol_ButtonHovered] = primary;
  colors[ImGuiCol_ButtonActive] = accent;

  colors[ImGuiCol_FrameBg] = ImVec4(1.0f, 1.0f, 1.0f, 0.04f);
  colors[ImGuiCol_FrameBgHovered] = ImVec4(1.0f, 1.0f, 1.0f, 0.10f);
  colors[ImGuiCol_FrameBgActive] = ImVec4(1.0f, 1.0f, 1.0f, 0.16f);

  colors[ImGuiCol_Tab] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
  colors[ImGuiCol_TabHovered] = primary;
  colors[ImGuiCol_TabActive] = secondary;
  colors[ImGuiCol_TabUnfocused] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
  colors[ImGuiCol_TabUnfocusedActive] = secondary;

  colors[ImGuiCol_TitleBg] = colors[ImGuiCol_WindowBg];
  colors[ImGuiCol_TitleBgActive] = colors[ImGuiCol_WindowBg];
  colors[ImGuiCol_TitleBgCollapsed] = colors[ImGuiCol_WindowBg];

  colors[ImGuiCol_PlotLines] = primary;
  colors[ImGuiCol_PlotLinesHovered] = accent;
  colors[ImGuiCol_PlotHistogram] = primary;
  colors[ImGuiCol_PlotHistogramHovered] = accent;

  colors[ImGuiCol_Text] = ImVec4(0.95f, 0.95f, 1.00f, 1.00f);
  colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
  
  colors[ImGuiCol_Separator] = colors[ImGuiCol_Border];
}

/// Apply a light theme variant.
inline void ApplyHafsLightTheme() {
  ImGui::StyleColorsLight();
  ImGuiStyle& style = ImGui::GetStyle();

  style.WindowRounding = 4.0f;
  style.FrameRounding = 2.0f;
  style.GrabRounding = 2.0f;

  ImVec4* colors = style.Colors;
  colors[ImGuiCol_PlotLines] = ImVec4(0.20f, 0.50f, 0.80f, 1.00f);
  colors[ImGuiCol_PlotHistogram] = ImVec4(0.20f, 0.70f, 0.50f, 1.00f);
}

}  // namespace afs::viz::themes
