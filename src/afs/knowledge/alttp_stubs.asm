; ALTTP Routine Stubs for asar validation
; Auto-generated list of common routine stubs
; These allow asar to validate code that calls these routines

lorom
warnings disable W1018

; =============================================================================
; Memory addresses / defines
; =============================================================================

; Link state
!LINK_HEALTH = $7EF36C
!LINK_MAX_HEALTH = $7EF36D
!LINK_X_LO = $7E0022
!LINK_X_HI = $7E0023
!LINK_Y_LO = $7E0020
!LINK_Y_HI = $7E0021
!LINK_LAYER = $7E00EE
!LINK_DIRECTION = $7E002F
!LINK_STATE = $7E005D
!LINK_ACTION = $7E0370

; Inventory
!RUPEE_COUNT = $7EF360
!BOMB_COUNT = $7EF343
!ARROW_COUNT = $7EF377
!MAGIC_POWER = $7EF36B
!SWORD_LEVEL = $7EF359

; Sprite table bases
!SPRITE_X_LO = $7E0D00
!SPRITE_X_HI = $7E0D10
!SPRITE_Y_LO = $7E0D20
!SPRITE_Y_HI = $7E0D30
!SPRITE_STATE = $7E0DD0
!SPRITE_TYPE = $7E0E20

; Game state
!GAME_MODE = $7E0010
!GAME_SUBMODE = $7E0011
!CURRENT_ROOM = $7E00A0
!WORLD_FLAG = $7E007B

; Controller
!JOYPAD_HELD = $7E00F0
!JOYPAD_PRESSED = $7E00F2

; Common sprite timer macros
!timer_0 = $0DF0
!timer_1 = $0E00
!timer_2 = $0E10
!timer_3 = $0E20
!timer_4 = $0E30

; Direction macros
!up = $00
!down = $02
!left = $04
!right = $06

; Message command macros
!CMD_F6_SCROLL = $F6
!CMD_F8_ROW2 = $F8
!CMD_F9_ROW3 = $F9
!CMD_FA_PAUSE = $FA
!CMD_FB_END = $FB
!CMD_FC_SPEED = $FC
!CMD_FD_KANJI = $FD
!CMD_FE_EXTCMD = $FE

; Color/palette
!color = $0000

; =============================================================================
; Sprite Routines (stubs)
; =============================================================================

org $008000

Sprite_SpawnDynamically:
    RTL

Sprite_SetSpawnedCoords:
    RTL

Sprite_DrawMultiple:
    RTL

Sprite_PrepAndDrawSingleLargeLong:
    RTL

Sprite_CheckIfActive:
    RTS

Sprite2_CheckIfActive:
    RTS

Sprite3_CheckIfActive:
    RTS

Sprite4_CheckIfActive:
    RTS

Sprite_Move:
    RTS

Sprite2_Move:
    RTS

Sprite3_Move:
    RTS

Sprite4_Move:
    RTS

Sprite_CheckDamageToPlayerSameLayerLong:
    RTL

Sprite_DrawShadowLong:
    RTL

Sprite_PrepOamCoord:
    RTS

Sprite2_PrepOamCoord:
    RTS

Sprite_PrepOamCoordLong:
    RTL

Sprite_PlayerCantPassThrough:
    RTS

Sprite2_DirectionToFacePlayer:
    RTS

Sprite3_DirectionToFacePlayer:
    RTS

Sprite_ApplySpeedTowardsPlayerLong:
    RTL

Sprite_CheckTileCollision:
    RTS

Sprite3_CheckTileCollision:
    RTS

Sprite3_CheckDamage:
    RTS

Sprite_ShowMessageUnconditional:
    RTS

Sprite_ShowSolicitedMessageIfPlayerFacing:
    RTS

; =============================================================================
; Sound Routines
; =============================================================================

Sound_SetSfx2PanLong:
    RTL

Sound_SetSfx3PanLong:
    RTL

; =============================================================================
; Utility Routines
; =============================================================================

GetRandomInt:
    RTL

UseImplicitRegIndexedLocalJumpTable:
    RTS

Link_ReceiveItem:
    RTL

DrawItem:
    RTS

; =============================================================================
; OAM / Graphics Routines
; =============================================================================

OAM_AllocateFromRegionA:
    RTS

OAM_AllocateFromRegionB:
    RTS

Ancilla_SetOam_XY:
    RTS

Ancilla_PrepOamCoord:
    RTS

; =============================================================================
; Common macros
; =============================================================================

macro CheckFlag(addr, flag)
    LDA <addr>
    AND.b #<flag>
endmacro

; =============================================================================
; Additional Sprite Routines
; =============================================================================

Sprite3_DrawMultiple:
    RTS

Sprite4_DrawMultiple:
    RTS

Sprite_CheckIfRecoiling:
    RTS

Sprite3_CheckIfRecoiling:
    RTS

Sprite_MoveAltitude:
    RTS

Sprite3_MoveAltitude:
    RTS

Sprite_DrawShadow:
    RTS

Sprite4_CheckTileCollision:
    RTS

Sprite_NullifyHookshotDrag:
    RTS

Sprite_CheckDamageFromPlayerLong:
    RTL

Sprite2_CheckDamage:
    RTS

Sprite4_CheckDamage:
    RTS

Sprite_ProjectSpeedTowardsEntityLong:
    RTL

Sprite_ProjectSpeedTowardsPlayerLong:
    RTL

Player_HaltDashAttackLong:
    RTL

SpritePrep_Bosses:
    RTS

; =============================================================================
; Ancilla / Garnish Routines
; =============================================================================

Ancilla_MoveHoriz:
    RTS

Ancilla_MoveVert:
    RTS

Garnish_PrepOamCoord:
    RTS

; =============================================================================
; Object Routines
; =============================================================================

Object_Draw3xN:
    RTS

; =============================================================================
; Filter / Effect Routines
; =============================================================================

Filter_Majorly_Whiten_Color:
    RTS

; =============================================================================
; Text Routines
; =============================================================================

Text_IgnoreCommand:
    RTS

; =============================================================================
; Common Local Labels (stub definitions)
; These are normally defined locally but we provide fallback stubs
; =============================================================================

BRANCH_ALPHA:
BRANCH_BETA:
BRANCH_GAMMA:
BRANCH_DELTA:
BRANCH_EPSILON:
BRANCH_:
    RTS

; Common sprite action variable
!SprAction = $0E30

; VMDATA for graphics
VMDATA:
    RTS

; === USER CODE BELOW ===
