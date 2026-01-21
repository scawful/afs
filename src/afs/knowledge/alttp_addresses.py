"""ALTTP (A Link to the Past) memory address reference data.

This module provides ground truth reference data for common ALTTP memory
addresses used in assembly code validation and training data generation.

Address Format:
- Bank $7E/$7F: WRAM (Work RAM) - mirrored at $0000-$1FFF in bank 00
- Bank $00-$3F: ROM and hardware registers
- Direct page addresses (< $0100) are relative to register D

Common conventions:
- $7E0000-$7E1FFF: Direct page accessible (mirrors $0000-$1FFF)
- $7E2000-$7EFFFF: General WRAM
- $7F0000-$7FFFFF: Extended WRAM (often used for buffers/tables)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AddressCategory(Enum):
    """Categories for ALTTP memory addresses."""
    LINK_STATE = "link_state"
    INVENTORY = "inventory"
    SPRITE = "sprite"
    MAP = "map"
    GRAPHICS = "graphics"
    SOUND = "sound"
    GAME_STATE = "game_state"
    HARDWARE = "hardware"
    MISC = "misc"


@dataclass
class AddressInfo:
    """Information about a memory address."""
    address: int | str  # Can be int or range string like "0D00-0D0F"
    name: str
    description: str
    category: AddressCategory
    size: int = 1  # Size in bytes
    bank: str = "7E"  # Default to WRAM bank
    read_write: str = "rw"  # "r", "w", or "rw"
    notes: str = ""

    @property
    def full_address(self) -> str:
        """Return full 24-bit address string."""
        if isinstance(self.address, str):
            return f"${self.bank}{self.address}"
        return f"${self.bank}{self.address:04X}"


# =============================================================================
# Link State Addresses
# =============================================================================

LINK_STATE_ADDRESSES: dict[str, AddressInfo] = {
    # Health
    "link_health": AddressInfo(
        address=0xF36C,
        name="Link Current Health",
        description="Link's current health. Each heart = 8 units. Half heart = 4 units.",
        category=AddressCategory.LINK_STATE,
        bank="7E",
        size=1,
        notes="Max value depends on heart containers. Full heart = 08, empty = 00.",
    ),
    "link_max_health": AddressInfo(
        address=0xF36D,
        name="Link Max Health",
        description="Link's maximum health (heart containers). Each container = 8 units.",
        category=AddressCategory.LINK_STATE,
        bank="7E",
        size=1,
        notes="Starting value is 0x18 (3 hearts). Max is 0xA0 (20 hearts).",
    ),

    # Position
    "link_x_coord": AddressInfo(
        address=0x0022,
        name="Link X Coordinate",
        description="Link's X coordinate on current screen.",
        category=AddressCategory.LINK_STATE,
        bank="7E",
        size=2,
        notes="16-bit value. Low byte at $22, high byte at $23.",
    ),
    "link_y_coord": AddressInfo(
        address=0x0020,
        name="Link Y Coordinate",
        description="Link's Y coordinate on current screen.",
        category=AddressCategory.LINK_STATE,
        bank="7E",
        size=2,
        notes="16-bit value. Low byte at $20, high byte at $21.",
    ),
    "link_layer": AddressInfo(
        address=0x00EE,
        name="Link Layer",
        description="Which BG layer Link is on (0 = upper, 1 = lower).",
        category=AddressCategory.LINK_STATE,
        bank="7E",
        size=1,
    ),

    # State
    "link_direction": AddressInfo(
        address=0x002F,
        name="Link Direction",
        description="Link's facing direction. 0=up, 2=down, 4=left, 6=right.",
        category=AddressCategory.LINK_STATE,
        bank="7E",
        size=1,
    ),
    "link_state": AddressInfo(
        address=0x005D,
        name="Link State",
        description="Link's current action state (walking, swimming, etc).",
        category=AddressCategory.LINK_STATE,
        bank="7E",
        size=1,
        notes="Complex state machine. See disassembly for full list.",
    ),
    "link_pose": AddressInfo(
        address=0x005E,
        name="Link Pose",
        description="Link's current pose/animation frame.",
        category=AddressCategory.LINK_STATE,
        bank="7E",
        size=1,
    ),
    "link_speed": AddressInfo(
        address=0x005E,
        name="Link Speed",
        description="Link's current movement speed.",
        category=AddressCategory.LINK_STATE,
        bank="7E",
        size=1,
    ),
    "link_invincibility": AddressInfo(
        address=0x031F,
        name="Link Invincibility Timer",
        description="Countdown timer for Link's invincibility frames.",
        category=AddressCategory.LINK_STATE,
        bank="7E",
        size=1,
        notes="Decrements each frame. Non-zero = invincible.",
    ),
}


# =============================================================================
# Inventory Addresses
# =============================================================================

INVENTORY_ADDRESSES: dict[str, AddressInfo] = {
    # Currency
    "rupee_count": AddressInfo(
        address=0xF360,
        name="Rupee Count",
        description="Current rupee count (16-bit).",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=2,
        notes="Low byte at $F360, high byte at $F361. Max 999.",
    ),
    "rupee_goal": AddressInfo(
        address=0xF362,
        name="Rupee Goal",
        description="Target rupee count for counter animation.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=2,
        notes="Counter animates toward this value.",
    ),

    # Bombs and Arrows
    "bomb_count": AddressInfo(
        address=0xF343,
        name="Bomb Count",
        description="Current number of bombs.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "bomb_max": AddressInfo(
        address=0xF370,
        name="Bomb Capacity",
        description="Maximum bomb capacity.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
        notes="Upgradeable. Default 10, max 50.",
    ),
    "arrow_count": AddressInfo(
        address=0xF377,
        name="Arrow Count",
        description="Current number of arrows.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "arrow_max": AddressInfo(
        address=0xF371,
        name="Arrow Capacity",
        description="Maximum arrow capacity.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
        notes="Upgradeable. Default 30, max 70.",
    ),

    # Magic
    "magic_power": AddressInfo(
        address=0xF36B,
        name="Magic Power",
        description="Current magic meter value.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
        notes="$80 = full, $00 = empty. Half magic doubles effective MP.",
    ),
    "magic_filler": AddressInfo(
        address=0xF373,
        name="Magic Filler",
        description="Amount of magic to restore (from pickups).",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),

    # Keys
    "small_key_count": AddressInfo(
        address=0xF36F,
        name="Small Key Count",
        description="Number of small keys for current dungeon.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),

    # Equipment flags
    "sword_level": AddressInfo(
        address=0xF359,
        name="Sword Level",
        description="Current sword. 0=none, 1=fighter, 2=master, 3=tempered, 4=gold.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "shield_level": AddressInfo(
        address=0xF35A,
        name="Shield Level",
        description="Current shield. 0=none, 1=fighter, 2=fire, 3=mirror.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "armor_level": AddressInfo(
        address=0xF35B,
        name="Armor Level",
        description="Current armor. 0=green, 1=blue, 2=red.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "glove_level": AddressInfo(
        address=0xF354,
        name="Glove Level",
        description="Current gloves. 0=none, 1=power, 2=titan.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),

    # Items
    "bow_type": AddressInfo(
        address=0xF340,
        name="Bow Type",
        description="Bow item. 0=none, 1=bow, 2=bow+arrows, 3=silver.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "boomerang_type": AddressInfo(
        address=0xF341,
        name="Boomerang Type",
        description="Boomerang. 0=none, 1=blue, 2=red.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "hookshot": AddressInfo(
        address=0xF342,
        name="Hookshot",
        description="Hookshot item. 0=no, 1=yes.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "mushroom_powder": AddressInfo(
        address=0xF344,
        name="Mushroom/Powder",
        description="Magic mushroom/powder. 0=none, 1=mushroom, 2=powder.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "fire_rod": AddressInfo(
        address=0xF345,
        name="Fire Rod",
        description="Fire rod item. 0=no, 1=yes.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "ice_rod": AddressInfo(
        address=0xF346,
        name="Ice Rod",
        description="Ice rod item. 0=no, 1=yes.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "bombos": AddressInfo(
        address=0xF347,
        name="Bombos Medallion",
        description="Bombos medallion. 0=no, 1=yes.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "ether": AddressInfo(
        address=0xF348,
        name="Ether Medallion",
        description="Ether medallion. 0=no, 1=yes.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "quake": AddressInfo(
        address=0xF349,
        name="Quake Medallion",
        description="Quake medallion. 0=no, 1=yes.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "lamp": AddressInfo(
        address=0xF34A,
        name="Lamp",
        description="Lamp item. 0=no, 1=yes.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "hammer": AddressInfo(
        address=0xF34B,
        name="Hammer",
        description="Magic hammer. 0=no, 1=yes.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "flute": AddressInfo(
        address=0xF34C,
        name="Flute",
        description="Flute/shovel. 0=none, 1=shovel, 2=flute, 3=flute+bird.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "net": AddressInfo(
        address=0xF34D,
        name="Bug Net",
        description="Bug catching net. 0=no, 1=yes.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "book": AddressInfo(
        address=0xF34E,
        name="Book of Mudora",
        description="Book of Mudora. 0=no, 1=yes.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "bottle_count": AddressInfo(
        address=0xF34F,
        name="Bottle Count",
        description="Number of bottles (0-4).",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "cane_somaria": AddressInfo(
        address=0xF350,
        name="Cane of Somaria",
        description="Cane of Somaria. 0=no, 1=yes.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "cane_byrna": AddressInfo(
        address=0xF351,
        name="Cane of Byrna",
        description="Cane of Byrna. 0=no, 1=yes.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "cape": AddressInfo(
        address=0xF352,
        name="Magic Cape",
        description="Magic cape. 0=no, 1=yes.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "mirror": AddressInfo(
        address=0xF353,
        name="Magic Mirror",
        description="Magic mirror. 0=no, 1=yes (2=scroll in JP).",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "boots": AddressInfo(
        address=0xF355,
        name="Pegasus Boots",
        description="Pegasus boots. 0=no, 1=yes.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "flippers": AddressInfo(
        address=0xF356,
        name="Zora Flippers",
        description="Zora's flippers. 0=no, 1=yes.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
    "moon_pearl": AddressInfo(
        address=0xF357,
        name="Moon Pearl",
        description="Moon pearl. 0=no, 1=yes.",
        category=AddressCategory.INVENTORY,
        bank="7E",
        size=1,
    ),
}


# =============================================================================
# Sprite Table Addresses (Direct Page - $0000-$0FFF range)
# =============================================================================

SPRITE_TABLES: dict[str, AddressInfo] = {
    # Sprite X position
    "sprite_x_lo": AddressInfo(
        address="0D00-0D0F",
        name="Sprite X Low Bytes",
        description="Low bytes of sprite X coordinates. One byte per sprite slot (0-15).",
        category=AddressCategory.SPRITE,
        bank="7E",
        size=16,
        notes="Indexed by sprite slot. Sprite 0 at $0D00, sprite 15 at $0D0F.",
    ),
    "sprite_x_hi": AddressInfo(
        address="0D10-0D1F",
        name="Sprite X High Bytes",
        description="High bytes of sprite X coordinates. One byte per sprite slot.",
        category=AddressCategory.SPRITE,
        bank="7E",
        size=16,
        notes="Combined with low byte: full_x = (hi << 8) | lo",
    ),

    # Sprite Y position
    "sprite_y_lo": AddressInfo(
        address="0D20-0D2F",
        name="Sprite Y Low Bytes",
        description="Low bytes of sprite Y coordinates. One byte per sprite slot.",
        category=AddressCategory.SPRITE,
        bank="7E",
        size=16,
    ),
    "sprite_y_hi": AddressInfo(
        address="0D30-0D3F",
        name="Sprite Y High Bytes",
        description="High bytes of sprite Y coordinates. One byte per sprite slot.",
        category=AddressCategory.SPRITE,
        bank="7E",
        size=16,
    ),

    # Sprite state
    "sprite_state": AddressInfo(
        address="0DD0-0DDF",
        name="Sprite State",
        description="Current state of each sprite. 0=inactive, various values for active states.",
        category=AddressCategory.SPRITE,
        bank="7E",
        size=16,
        notes="$00=dead/inactive, $09=active, $0A=falling into pit, etc.",
    ),
    "sprite_type": AddressInfo(
        address="0E20-0E2F",
        name="Sprite Type",
        description="Sprite type/ID for each slot. Determines sprite behavior.",
        category=AddressCategory.SPRITE,
        bank="7E",
        size=16,
        notes="See sprite ID table in disassembly.",
    ),
    "sprite_subtype": AddressInfo(
        address="0E30-0E3F",
        name="Sprite Subtype",
        description="Sprite subtype/variant. Modifies sprite behavior.",
        category=AddressCategory.SPRITE,
        bank="7E",
        size=16,
    ),

    # Sprite health
    "sprite_health": AddressInfo(
        address="0E50-0E5F",
        name="Sprite Health",
        description="Health/HP of each sprite.",
        category=AddressCategory.SPRITE,
        bank="7E",
        size=16,
        notes="Decrements when hit. Sprite dies when reaching 0.",
    ),

    # Sprite movement
    "sprite_x_speed": AddressInfo(
        address="0D40-0D4F",
        name="Sprite X Speed",
        description="Horizontal velocity of each sprite (signed).",
        category=AddressCategory.SPRITE,
        bank="7E",
        size=16,
        notes="Signed value. Positive = right, negative = left.",
    ),
    "sprite_y_speed": AddressInfo(
        address="0D50-0D5F",
        name="Sprite Y Speed",
        description="Vertical velocity of each sprite (signed).",
        category=AddressCategory.SPRITE,
        bank="7E",
        size=16,
        notes="Signed value. Positive = down, negative = up.",
    ),
    "sprite_z_speed": AddressInfo(
        address="0D60-0D6F",
        name="Sprite Z Speed",
        description="Z-axis (height) velocity for jumping/falling.",
        category=AddressCategory.SPRITE,
        bank="7E",
        size=16,
    ),
    "sprite_z_height": AddressInfo(
        address="0F00-0F0F",
        name="Sprite Z Height",
        description="Current Z height (shadow offset) of each sprite.",
        category=AddressCategory.SPRITE,
        bank="7E",
        size=16,
    ),

    # Sprite timers and flags
    "sprite_timer": AddressInfo(
        address="0DF0-0DFF",
        name="Sprite Timer 1",
        description="General purpose timer for each sprite.",
        category=AddressCategory.SPRITE,
        bank="7E",
        size=16,
    ),
    "sprite_timer2": AddressInfo(
        address="0E00-0E0F",
        name="Sprite Timer 2",
        description="Secondary timer for each sprite.",
        category=AddressCategory.SPRITE,
        bank="7E",
        size=16,
    ),
    "sprite_stun_timer": AddressInfo(
        address="0E70-0E7F",
        name="Sprite Stun Timer",
        description="Stun/freeze timer for each sprite.",
        category=AddressCategory.SPRITE,
        bank="7E",
        size=16,
        notes="Countdown. Sprite cannot act while > 0.",
    ),
    "sprite_recoil_timer": AddressInfo(
        address="0E60-0E6F",
        name="Sprite Recoil Timer",
        description="Recoil/knockback timer for each sprite.",
        category=AddressCategory.SPRITE,
        bank="7E",
        size=16,
    ),

    # Sprite graphics
    "sprite_oam_props": AddressInfo(
        address="0E40-0E4F",
        name="Sprite OAM Properties",
        description="OAM properties (palette, priority, flip) for each sprite.",
        category=AddressCategory.SPRITE,
        bank="7E",
        size=16,
    ),
    "sprite_layer": AddressInfo(
        address="0F20-0F2F",
        name="Sprite Layer",
        description="Which BG layer the sprite is on.",
        category=AddressCategory.SPRITE,
        bank="7E",
        size=16,
    ),

    # Sprite flags
    "sprite_bump_damage": AddressInfo(
        address="0CD2",
        name="Sprite Bump Damage Class",
        description="Table of damage values for sprite collision.",
        category=AddressCategory.SPRITE,
        bank="7E",
        size=16,
        notes="Indexed by sprite class, not slot.",
    ),
    "sprite_persist_flag": AddressInfo(
        address="0B58-0B67",
        name="Sprite Persistence Flag",
        description="Whether sprite should respawn on room reload.",
        category=AddressCategory.SPRITE,
        bank="7E",
        size=16,
    ),
}


# =============================================================================
# WRAM General Addresses
# =============================================================================

WRAM_ADDRESSES: dict[str, AddressInfo] = {
    # Game mode
    "game_mode": AddressInfo(
        address=0x0010,
        name="Game Mode",
        description="Current game mode/module. Determines main game state.",
        category=AddressCategory.GAME_STATE,
        bank="7E",
        size=1,
        notes="$07=dungeon, $09=overworld, $0E=text/dialog, etc.",
    ),
    "game_submode": AddressInfo(
        address=0x0011,
        name="Game Submode",
        description="Sub-state within current game mode.",
        category=AddressCategory.GAME_STATE,
        bank="7E",
        size=1,
    ),
    "game_subsubmode": AddressInfo(
        address=0x00B0,
        name="Game Sub-Submode",
        description="Tertiary state for complex transitions.",
        category=AddressCategory.GAME_STATE,
        bank="7E",
        size=1,
    ),

    # Frame counter
    "frame_counter": AddressInfo(
        address=0x001A,
        name="Frame Counter",
        description="Increments every frame. Used for timing and animation.",
        category=AddressCategory.GAME_STATE,
        bank="7E",
        size=1,
        notes="Wraps at 256. Two-byte counter at $1A-$1B.",
    ),

    # Controller input
    "joypad1_held": AddressInfo(
        address=0x00F0,
        name="Joypad 1 Held",
        description="Buttons currently held on controller 1.",
        category=AddressCategory.HARDWARE,
        bank="7E",
        size=2,
        notes="Standard SNES button mapping. Low byte at $F0.",
    ),
    "joypad1_pressed": AddressInfo(
        address=0x00F2,
        name="Joypad 1 Pressed",
        description="Buttons newly pressed this frame on controller 1.",
        category=AddressCategory.HARDWARE,
        bank="7E",
        size=2,
        notes="Set for one frame on button press.",
    ),
    "joypad1_filtered": AddressInfo(
        address=0x00F4,
        name="Joypad 1 Filtered",
        description="Filtered joypad input (respects input locks).",
        category=AddressCategory.HARDWARE,
        bank="7E",
        size=2,
    ),

    # Screen/room
    "current_room": AddressInfo(
        address=0x00A0,
        name="Current Room ID",
        description="Current dungeon room or overworld area ID.",
        category=AddressCategory.MAP,
        bank="7E",
        size=2,
        notes="16-bit room number.",
    ),
    "overworld_area": AddressInfo(
        address=0x008A,
        name="Overworld Area",
        description="Current overworld screen/area index.",
        category=AddressCategory.MAP,
        bank="7E",
        size=1,
    ),
    "dungeon_id": AddressInfo(
        address=0x040C,
        name="Dungeon ID",
        description="Current dungeon number (0=sewers, 1=HC, etc).",
        category=AddressCategory.MAP,
        bank="7E",
        size=1,
    ),
    "world_flag": AddressInfo(
        address=0x007B,
        name="World Flag",
        description="Light world (0) or Dark world (1) flag.",
        category=AddressCategory.MAP,
        bank="7E",
        size=1,
    ),

    # Screen scrolling
    "bg1_x_scroll": AddressInfo(
        address=0x00E0,
        name="BG1 X Scroll",
        description="Background 1 horizontal scroll position.",
        category=AddressCategory.GRAPHICS,
        bank="7E",
        size=2,
    ),
    "bg1_y_scroll": AddressInfo(
        address=0x00E2,
        name="BG1 Y Scroll",
        description="Background 1 vertical scroll position.",
        category=AddressCategory.GRAPHICS,
        bank="7E",
        size=2,
    ),
    "bg2_x_scroll": AddressInfo(
        address=0x00E4,
        name="BG2 X Scroll",
        description="Background 2 horizontal scroll position.",
        category=AddressCategory.GRAPHICS,
        bank="7E",
        size=2,
    ),
    "bg2_y_scroll": AddressInfo(
        address=0x00E6,
        name="BG2 Y Scroll",
        description="Background 2 vertical scroll position.",
        category=AddressCategory.GRAPHICS,
        bank="7E",
        size=2,
    ),
    "camera_x": AddressInfo(
        address=0x00E2,
        name="Camera X",
        description="Camera X position (same as BG scroll in most cases).",
        category=AddressCategory.GRAPHICS,
        bank="7E",
        size=2,
    ),
    "camera_y": AddressInfo(
        address=0x00E8,
        name="Camera Y",
        description="Camera Y position.",
        category=AddressCategory.GRAPHICS,
        bank="7E",
        size=2,
    ),

    # RNG
    "rng_value": AddressInfo(
        address=0x002C,
        name="RNG Value",
        description="Random number generator state.",
        category=AddressCategory.MISC,
        bank="7E",
        size=2,
        notes="16-bit LFSR used for randomness.",
    ),

    # Music and SFX
    "music_control": AddressInfo(
        address=0x0132,
        name="Music Control",
        description="Music track to play. Written to trigger music change.",
        category=AddressCategory.SOUND,
        bank="7E",
        size=1,
    ),
    "sfx_1": AddressInfo(
        address=0x012E,
        name="SFX Channel 1",
        description="Sound effect to play on channel 1.",
        category=AddressCategory.SOUND,
        bank="7E",
        size=1,
    ),
    "sfx_2": AddressInfo(
        address=0x012F,
        name="SFX Channel 2",
        description="Sound effect to play on channel 2.",
        category=AddressCategory.SOUND,
        bank="7E",
        size=1,
    ),
    "sfx_3": AddressInfo(
        address=0x0130,
        name="SFX Channel 3",
        description="Sound effect to play on channel 3.",
        category=AddressCategory.SOUND,
        bank="7E",
        size=1,
    ),

    # Pause and menu
    "pause_flag": AddressInfo(
        address=0x00E3,
        name="Pause Flag",
        description="Whether game is paused.",
        category=AddressCategory.GAME_STATE,
        bank="7E",
        size=1,
    ),
    "menu_cursor": AddressInfo(
        address=0x0205,
        name="Menu Cursor Position",
        description="Current cursor position in menus.",
        category=AddressCategory.GAME_STATE,
        bank="7E",
        size=1,
    ),

    # NMI and IRQ
    "nmi_flag": AddressInfo(
        address=0x0016,
        name="NMI Flag",
        description="Set by NMI handler, cleared by main loop.",
        category=AddressCategory.HARDWARE,
        bank="7E",
        size=1,
        notes="Used for frame synchronization.",
    ),
}


# =============================================================================
# Combined Address Dictionary
# =============================================================================

ALTTP_ADDRESSES: dict[str, AddressInfo] = {
    **LINK_STATE_ADDRESSES,
    **INVENTORY_ADDRESSES,
    **SPRITE_TABLES,
    **WRAM_ADDRESSES,
}


# =============================================================================
# Utility Functions
# =============================================================================

def get_address_info(name: str) -> AddressInfo | None:
    """Get address info by name."""
    return ALTTP_ADDRESSES.get(name)


def get_addresses_by_category(category: AddressCategory) -> dict[str, AddressInfo]:
    """Get all addresses in a category."""
    return {
        name: info
        for name, info in ALTTP_ADDRESSES.items()
        if info.category == category
    }


def format_address_reference(name: str, include_notes: bool = False) -> str:
    """Format an address as a reference string for assembly comments.

    Example output:
        ; $7EF36C: Link Current Health - Link's current health. Each heart = 8 units.
    """
    info = ALTTP_ADDRESSES.get(name)
    if not info:
        return f"; Unknown address: {name}"

    parts = [f"; {info.full_address}: {info.name}"]
    if info.description:
        parts.append(f" - {info.description}")
    if include_notes and info.notes:
        parts.append(f" ({info.notes})")

    return "".join(parts)


def lookup_by_address(address: int, bank: str = "7E") -> list[tuple[str, AddressInfo]]:
    """Find address info by numeric address value.

    Returns list of (name, info) tuples that match the address.
    """
    results = []
    for name, info in ALTTP_ADDRESSES.items():
        if info.bank != bank:
            continue

        if isinstance(info.address, int):
            if info.address == address:
                results.append((name, info))
            elif info.size > 1 and info.address <= address < info.address + info.size:
                results.append((name, info))
        elif isinstance(info.address, str) and "-" in info.address:
            try:
                start_str, end_str = info.address.split("-")
                start = int(start_str, 16)
                end = int(end_str, 16)
                if start <= address <= end:
                    results.append((name, info))
            except ValueError:
                continue

    return results


def generate_include_header() -> str:
    """Generate an asar include file with address defines.

    Returns asar-compatible assembly code defining common addresses.
    """
    lines = [
        "; ALTTP Memory Address Definitions",
        "; Auto-generated from alttp_addresses.py",
        "; For use with asar assembler",
        ";",
        "",
    ]

    categories = [
        (AddressCategory.LINK_STATE, "Link State"),
        (AddressCategory.INVENTORY, "Inventory"),
        (AddressCategory.SPRITE, "Sprite Tables"),
        (AddressCategory.GAME_STATE, "Game State"),
        (AddressCategory.MAP, "Map/Room"),
        (AddressCategory.GRAPHICS, "Graphics"),
        (AddressCategory.SOUND, "Sound"),
        (AddressCategory.HARDWARE, "Hardware"),
    ]

    for category, title in categories:
        addrs = get_addresses_by_category(category)
        if not addrs:
            continue

        lines.append(f"; === {title} ===")
        for name, info in sorted(addrs.items()):
            # Skip range addresses for defines (they're tables)
            if isinstance(info.address, str):
                lines.append(f"; {name}: ${info.bank}{info.address} (table)")
                continue

            define_name = name.upper().replace(" ", "_")
            lines.append(f"!{define_name} = ${info.bank}{info.address:04X}")

        lines.append("")

    return "\n".join(lines)
