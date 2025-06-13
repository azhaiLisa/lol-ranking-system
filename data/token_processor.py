import json
from typing import List, Dict, Any
from itertools import chain
import glob

# Find all matching files if used for batched processing
# json_files = glob.glob("match_ranked_*.json")
MATCH_FILE = "matches.json"
OUTPUT_FILE = "processed_tokens.txt"

# Constants for token mapping
SKILL_MAP = {1: "Q", 2: "W", 3: "E", 4: "R"}
WARD_TYPE_MAP = {
    "YELLOW_TRINKET": "YELLOW",
    "CONTROL_WARD": "CONTROL",
    "SIGHT_WARD": "SIGHT",
    "BLUE_TRINKET": "BLUE",
    "UNDEFINED": "UNDEFINED"
}

# Item category mapping
ITEM_CATEGORY_MAP = {
    "1001": "BOOTS",
    "1004": "FAERIE_CHARM",
    "1006": "REJUVENATION_BEAD",
    "1011": "GIANTS_BELT",
    "1018": "CLOAK_OF_AGILITY",
    "1026": "BLASTING_WAND",
    "1027": "SAPPHIRE_CRYSTAL",
    "1028": "RUBY_CRYSTAL",
    "1029": "CLOTH_ARMOR",
    "1031": "CHAIN_VEST",
    "1033": "NULL_MAGIC_MANTLE",
    "1035": "EMBERKNIFE",
    "1036": "LONG_SWORD",
    "1037": "PICKAXE",
    "1038": "B_F_SWORD",
    "1039": "HAILBLADE",
    "1040": "OBSIDIAN_EDGE",
    "1042": "DAGGER",
    "1043": "RECURVE_BOW",
    "1052": "AMPLIFYING_TOME",
    "1053": "VAMPIRIC_SCEPTER",
    "1054": "DORANS_SHIELD",
    "1055": "DORANS_BLADE",
    "1056": "DORANS_RING",
    "1057": "NEGATRON_CLOAK",
    "1058": "NEEDLESSLY_LARGE_ROD",
    "1082": "DARK_SEAL",
    "1083": "CULL",
    "1101": "SCORCHCLAW_PUP",
    "1102": "GUSTWALKER_HATCHLING",
    "1103": "MOSSTOMPER_SEEDLING",
    "1104": "EYE_OF_THE_HERALD",
    "1105": "MOSSTOMPER_SEEDLING",
    "1106": "GUSTWALKER_HATCHLING",
    "1107": "SCORCHCLAW_PUP",
    "1500": "PENETRATING_BULLETS",
    "1501": "FORTIFICATION",
    "1502": "REINFORCED_ARMOR",
    "1503": "WARDENS_EYE",
    "1504": "VANGUARD",
    "1506": "REINFORCED_ARMOR",
    "1507": "OVERCHARGED",
    "1508": "ANTI_TOWER_SOCKS",
    "1509": "GUSTO",
    "1510": "PHREAKISH_GUSTO",
    "1511": "SUPER_MECH_ARMOR",
    "1512": "SUPER_MECH_POWER_FIELD",
    "1515": "TURRET_PLATING",
    "1516": "STRUCTURE_BOUNTY",
    "1517": "STRUCTURE_BOUNTY",
    "1518": "STRUCTURE_BOUNTY",
    "1519": "STRUCTURE_BOUNTY",
    "1520": "OVERERCHARGEDHA",
    "1521": "FORTIFICATION",
    "1522": "TOWER_POWER_UP",
    "1523": "OVERCHARGED",
    "2003": "HEALTH_POTION",
    "2010": "TOTAL_BISCUIT_OF_EVERLASTING_WILL",
    "2015": "KIRCHEIS_SHARD",
    "2019": "STEEL_SIGIL",
    "2020": "THE_BRUTALIZER",
    "2021": "TUNNELER",
    "2022": "GLOWING_MOTE",
    "2031": "REFILLABLE_POTION",
    "2033": "CORRUPTING_POTION",
    "2049": "GUARDIANS_AMULET",
    "2050": "GUARDIANS_SHROUD",
    "2051": "GUARDIANS_HORN",
    "2052": "PORO_SNAX",
    "2055": "CONTROL_WARD",
    "2056": "STEALTH_WARD",
    "2065": "SHURELYAS_BATTLESONG",
    "2138": "ELIXIR_OF_IRON",
    "2139": "ELIXIR_OF_SORCERY",
    "2140": "ELIXIR_OF_WRATH",
    "2141": "CAPPA_JUICE",
    "2142": "JUICE_OF_POWER",
    "2143": "JUICE_OF_VITALITY",
    "2144": "JUICE_OF_HASTE",
    "2145": "LUCKY_DICE",
    "2146": "ENHANCED_LUCKY_DICE",
    "2150": "ELIXIR_OF_SKILL",
    "2151": "ELIXIR_OF_AVARICE",
    "2152": "ELIXIR_OF_FORCE",
    "2403": "MINION_DEMATERIALIZER",
    "2420": "SEEKERS_ARMGUARD",
    "2421": "SHATTERED_ARMGUARD",
    "2422": "SLIGHTLY_MAGICAL_FOOTWEAR",
    "2501": "OVERLORDS_BLOODMAIL",
    "2502": "UNENDING_DESPAIR",
    "2503": "BLACKFIRE_TORCH",
    "2504": "KAENIC_ROOKERN",
    "2508": "FATED_ASHES",
    "3001": "EVENSHROUD",
    "3002": "TRAILBLAZER",
    "3003": "ARCHANGELS_STAFF",
    "3004": "MANAMUNE",
    "3005": "GHOSTCRAWLERS",
    "3006": "BERSERKERS_GREAVES",
    "3009": "BOOTS_OF_SWIFTNESS",
    "3010": "SYMBIOTIC_SOLES",
    "3011": "CHEMTECH_PUTRIFIER",
    "3012": "CHALICE_OF_BLESSING",
    "3013": "SYNCHRONIZED_SOULS",
    "3020": "SORCERERS_SHOES",
    "3023": "LIFEWELL_PENDANT",
    "3024": "GLACIAL_BUCKLER",
    "3026": "GUARDIAN_ANGEL",
    "3031": "INFINITY_EDGE",
    "3032": "YUN_TAL_WILDARROWS",
    "3033": "MORTAL_REMINDER",
    "3035": "LAST_WHISPER",
    "3036": "LORD_DOMINIKS_REGARDS",
    "3039": "ATMAS_RECKONING",
    "3040": "SERAPHS_EMBRACE",
    "3041": "MEJAIS_SOULSTEALER",
    "3042": "MURAMANA",
    "3044": "PHAGE",
    "3046": "PHANTOM_DANCER",
    "3047": "PLATED_STEELCAPS",
    "3050": "ZEKES_CONVERGENCE",
    "3051": "HEARTHBOUND_AXE",
    "3053": "STERAKS_GAGE",
    "3057": "SHEEN",
    "3065": "SPIRIT_VISAGE",
    "3066": "WINGED_MOONPLATE",
    "3067": "KINDLEGEM",
    "3068": "SUNFIRE_AEGIS",
    "3070": "TEAR_OF_THE_GODDESS",
    "3071": "BLACK_CLEAVER",
    "3072": "BLOODTHIRSTER",
    "3073": "EXPERIMENTAL_HEXPLATE",
    "3074": "RAVENOUS_HYDRA",
    "3075": "THORNMAIL",
    "3076": "BRAMBLE_VEST",
    "3077": "TIAMAT",
    "3078": "TRINITY_FORCE",
    "3082": "WARDENS_MAIL",
    "3083": "WARMOGS_ARMOR",
    "3084": "HEARTSTEEL",
    "3085": "RUNAANS_HURRICANE",
    "3086": "ZEAL",
    "3087": "STATIKK_SHIV",
    "3089": "RABADONS_DEATHCAP",
    "3091": "WITS_END",
    "3094": "RAPID_FIRECANNON",
    "3095": "STORMRAZOR",
    "3100": "LICH_BANE",
    "3102": "BANSHEES_VEIL",
    "3105": "AEGIS_OF_THE_LEGION",
    "3107": "REDEMPTION",
    "3108": "FIENDISH_CODEX",
    "3109": "KNIGHTS_VOW",
    "3110": "FROZEN_HEART",
    "3111": "MERCURYS_TREADS",
    "3112": "GUARDIANS_ORB",
    "3113": "AETHER_WISP",
    "3114": "FORBIDDEN_IDOL",
    "3115": "NASHORS_TOOTH",
    "3116": "RYLAIS_CRYSTAL_SCEPTER",
    "3117": "MOBILITY_BOOTS",
    "3118": "MALIGNANCE",
    "3119": "WINTERS_APPROACH",
    "3121": "FIMBULWINTER",
    "3123": "EXECUTIONERS_CALLING",
    "3124": "GUINSOOS_RAGEBLADE",
    "3128": "DEATHFIRE_GRASP",
    "3131": "SWORD_OF_THE_DIVINE",
    "3133": "CAULFIELDS_WARHAMMER",
    "3134": "SERRATED_DIRK",
    "3135": "VOID_STAFF",
    "3137": "CRYPTBLOOM",
    "3139": "MERCURIAL_SCIMITAR",
    "3140": "QUICKSILVER_SASH",
    "3142": "YOUMUUS_GHOSTBLADE",
    "3143": "RANDUINS_OMEN",
    "3144": "SCOUTS_SLINGSHOT",
    "3145": "HEXTECH_ALTERNATOR",
    "3146": "HEXTECH_GUNBLADE",
    "3147": "HAUNTING_GUISE",
    "3152": "HEXTECH_ROCKETBELT",
    "3153": "BLADE_OF_THE_RUINED_KING",
    "3155": "HEXDRINKER",
    "3156": "MAW_OF_MALMORTIUS",
    "3157": "ZHONYAS_HOURGLASS",
    "3158": "IONIAN_BOOTS_OF_LUCIDITY",
    "3161": "SPEAR_OF_SHOJIN",
    "3165": "MORELLONOMICON",
    "3170": "SWIFTMARCH",
    "3171": "CRIMSON_LUCIDITY",
    "3172": "GUNMETAL_GREAVES",
    "3173": "CHAINLACED_CRUSHERS",
    "3174": "ARMORED_ADVANCE",
    "3175": "SPELLSLINGERS_SHOES",
    "3176": "FOREVER_FORWARD",
    "3177": "GUARDIANS_BLADE",
    "3179": "UMBRAL_GLAIVE",
    "3181": "HULLBREAKER",
    "3184": "GUARDIANS_HAMMER",
    "3190": "LOCKET_OF_THE_IRON_SOLARI",
    "3193": "GARGOYLE_STONEPLATE",
    "3211": "SPECTRES_COWL",
    "3222": "MIKAELS_BLESSING",
    "3302": "TERMINUS",
    "3330": "SCARECROW_EFFIGY",
    "3340": "STEALTH_WARD",
    "3348": "ARCANE_SWEEPER",
    "3349": "LUCENT_SINGULARITY",
    "3363": "FARSIGHT_ALTERATION",
    "3364": "ORACLE_LENS",
    "3398": "SMALL_PARTY_FAVOR",
    "3399": "PARTY_FAVOR",
    "3400": "YOUR_CUT",
    "3430": "RITE_OF_RUIN",
    "3504": "ARDENT_CENSER",
    "3508": "ESSENCE_REAVER",
    "3513": "EYE_OF_THE_HERALD",
    "3599": "KALISTAS_BLACK_SPEAR",
    "3600": "KALISTAS_BLACK_SPEAR",
    "3742": "DEAD_MANS_PLATE",
    "3748": "TITANIC_HYDRA",
    "3801": "CRYSTALLINE_BRACER",
    "3802": "LOST_CHAPTER",
    "3803": "CATALYST_OF_AEONS",
    "3814": "EDGE_OF_NIGHT",
    "3850": "SPELLTHIEFS_EDGE",
    "3851": "FROSTFANG",
    "3853": "SHARD_OF_TRUE_ICE",
    "3854": "STEEL_SHOULDERGUARDS",
    "3855": "RUNESTEEL_SPAULDERS",
    "3857": "PAULDRONS_OF_WHITEROCK",
    "3858": "RELIC_SHIELD",
    "3859": "TARGONS_BUCKLER",
    "3860": "BULWARK_OF_THE_MOUNTAIN",
    "3862": "SPECTRAL_SICKLE",
    "3863": "HARROWING_CRESCENT",
    "3864": "BLACK_MIST_SCYTHE",
    "3865": "WORLD_ATLAS",
    "3866": "RUNIC_COMPASS",
    "3867": "BOUNTY_OF_WORLDS",
    "3869": "CELESTIAL_OPPOSITION",
    "3870": "DREAM_MAKER",
    "3871": "ZAZZAKS_REALMSPIKE",
    "3876": "SOLSTICE_SLEIGH",
    "3877": "BLOODSONG",
    "3901": "FIRE_AT_WILL",
    "3902": "DEATHS_DAUGHTER",
    "3903": "RAISE_MORALE",
    "3916": "OBLIVION_ORB",
    "4003": "LIFELINE",
    "4004": "SPECTRAL_CUTLASS",
    "4005": "IMPERIAL_MANDATE",
    "4010": "BLOODLETTERS_CURSE",
    "4011": "SWORD_OF_BLOSSOMING_DAWN",
    "4012": "SIN_EATER",
    "4013": "LIGHTNING_BRAID",
    "4014": "FROZEN_MALLET",
    "4015": "PERPLEXITY",
    "4016": "WORDLESS_PROMISE",
    "4017": "HELLFIRE_HATCHET",
    "4401": "FORCE_OF_NATURE",
    "4402": "INNERVATING_LOCKET",
    "4403": "THE_GOLDEN_SPATULA",
    "4628": "HORIZON_FOCUS",
    "4629": "COSMIC_DRIVE",
    "4630": "BLIGHTING_JEWEL",
    "4632": "VERDANT_BARRIER",
    "4633": "RIFTMAKER",
    "4635": "LEECHING_LEER",
    "4636": "NIGHT_HARVESTER",
    "4637": "DEMONIC_EMBRACE",
    "4638": "WATCHFUL_WARDSTONE",
    "4641": "STIRRING_WARDSTONE",
    "4642": "BANDLEGLASS_MIRROR",
    "4643": "VIGILANT_WARDSTONE",
    "4644": "CROWN_OF_THE_SHATTERED_QUEEN",
    "4645": "SHADOWFLAME",
    "4646": "STORMSURGE",
    "6029": "IRONSPIKE_WHIP",
    "6035": "SILVERMERE_DAWN",
    "6333": "DEATHS_DANCE",
    "6609": "CHEMPUNK_CHAINSWORD",
    "6610": "SUNDERED_SKY",
    "6616": "STAFF_OF_FLOWING_WATER",
    "6617": "MOONSTONE_RENEWER",
    "6620": "ECHOES_OF_HELIA",
    "6621": "DAWNCORE",
    "6630": "GOREDRINKER",
    "6631": "STRIDEBREAKER",
    "6632": "DIVINE_SUNDERER",
    "6653": "LIANDRYS_TORMENT",
    "6655": "LUDENS_COMPANION",
    "6656": "EVERFROST",
    "6657": "ROD_OF_AGES",
    "6660": "BAMIS_CINDER",
    "6662": "ICEBORN_GAUNTLET",
    "6664": "HOLLOW_RADIANCE",
    "6665": "JAKSHO,_THE_PROTEAN",
    "6667": "RADIANT_VIRTUE",
    "6670": "NOONQUIVER",
    "6671": "GALEFORCE",
    "6672": "KRAKEN_SLAYER",
    "6673": "IMMORTAL_SHIELDBOW",
    "6675": "NAVORI_FLICKERBLADE",
    "6676": "THE_COLLECTOR",
    "6677": "RAGEKNIFE",
    "6690": "RECTRIX",
    "6691": "DUSKBLADE_OF_DRAKTHARR",
    "6692": "ECLIPSE",
    "6693": "PROWLERS_CLAW",
    "6694": "SERYLDAS_GRUDGE",
    "6695": "SERPENTS_FANG",
    "6696": "AXIOM_ARC",
    "6697": "HUBRIS",
    "6698": "PROFANE_HYDRA",
    "6699": "VOLTAIC_CYCLOSWORD",
    "6700": "SHIELD_OF_THE_RAKKOR",
    "6701": "OPPORTUNITY",
    "7050": "GANGPLANK_PLACEHOLDER",
    "8001": "ANATHEMAS_CHAINS",
    "8010": "BLOODLETTERS_CURSE",
    "8020": "ABYSSAL_MASK",
    "9168": "LOCKED_WEAPON_SLOT",
    "9171": "CYCLONIC_SLICERS",
    "9172": "YUUMIBOT",
    "9173": "RADIANT_FIELD",
    "9174": "STATIKK_SWORD",
    "9175": "LIONESSS_LAMENT",
    "9176": "GATLING_BUNNY_GUNS",
    "9177": "SEARING_SHORTBOW",
    "9178": "THE_ANNIHILATOR",
    "9179": "BATTLE_BUNNY_CROSSBOW",
    "9180": "UWU_BLASTER",
    "9181": "VORTEX_GLOVE",
    "9183": "BLADE_O_RANG",
    "9184": "BUNNY_MEGA_BLAST",
    "9185": "ANTI_SHARK_SEA_MINE",
    "9187": "TIBBERS",
    "9188": "ANI_MINES",
    "9189": "FINAL_CITY_TRANSIT",
    "9190": "ECHOING_BATBLADES",
    "9192": "PAW_PRINT_POISONER",
    "9193": "ICEBLAST_ARMOR",
    "9271": "UNCEASING_CYCLONE",
    "9272": "YUUMIBOT_FINAL_FINAL",
    "9273": "EXPLOSIVE_EMBRACE",
    "9274": "PRUMBISS_ELECTROCARVER",
    "9275": "ENVELOPING_LIGHT",
    "9276": "DOUBLE_BUN_BUN_BARRAGE",
    "9277": "EVOLVED_EMBERSHOT",
    "9278": "ANIMAPOCALYPSE",
    "9279": "BUNNY_PRIME_BALLISTA",
    "9280": "OWO_BLASTER",
    "9281": "TEMPESTS_GAUNTLET",
    "9283": "QUAD_O_RANG",
    "9284": "RAPID_RABBIT_RAINDOWN",
    "9285": "NEVERENDING_MOBSTOMPER",
    "9287": "TIBBERS_(BEEG_EDITION)",
    "9288": "JINXS_TRI_NAMITE",
    "9289": "FC_LIMITED_EXPRESS",
    "9290": "VAYNES_CHROMABLADES",
    "9292": "BEARFOOT_CHEM_DISPENSER",
    "9293": "DEEP_FREEZE",
    "9300": "MEOW_MEOW",
    "9301": "SHIELD_SLAM",
    "9302": "SOUND_WAVE",
    "9303": "PILLORY_SWIPE",
    "9304": "STEEL_TEMPEST",
    "9305": "TENTACLE_SLAM",
    "9306": "WINGED_DAGGER",
    "9307": "GUIDING_HEX",
    "9308": "BUNNY_HOP",
    "9400": "BATTLE_CAT_BARRAGE",
    "9401": "LIGHT_OF_THE_LION",
    "9402": "ANIMA_ECHO",
    "9403": "SAVAGE_SLICE",
    "9404": "WANDERING_STORMS",
    "9405": "GRIZZLY_SMASH",
    "9406": "LOVERS_RICOCHET",
    "9407": "HOPPED_UP_HEX",
    "9408": "CARROT_CRASH",
    "126697": "HUBRIS",
    "220000": "STAT_BONUS",
    "220001": "LEGENDARY_FIGHTER_ITEM",
    "220002": "LEGENDARY_MARKSMAN_ITEM",
    "220003": "LEGENDARY_ASSASSIN_ITEM",
    "220004": "LEGENDARY_MAGE_ITEM",
    "220005": "LEGENDARY_TANK_ITEM",
    "220006": "LEGENDARY_SUPPORT_ITEM",
    "220007": "PRISMATIC_ITEM",
    "220008": "ANVIL_VOUCHER",
    "220009": "GOLD_STAT_ANVIL_VOUCHER",
    "220010": "PRISMATIC_STAT_VOUCHER",
    "220011": "BRAVERY_VOUCHER",
    "221011": "GIANTS_BELT",
    "221026": "BLASTING_WAND",
    "221031": "CHAIN_VEST",
    "221038": "B_F_SWORD",
    "221043": "RECURVE_BOW",
    "221053": "VAMPIRIC_SCEPTER",
    "221057": "NEGATRON_CLOAK",
    "221058": "NEEDLESSLY_LARGE_ROD",
    "222022": "GLOWING_MOTE",
    "222051": "GUARDIANS_HORN",
    "222065": "SHURELYAS_BATTLESONG",
    "222141": "CAPPA_JUICE",
    "222502": "UNENDING_DESPAIR",
    "222503": "BLACKFIRE_TORCH",
    "222504": "KAENIC_ROOKERN",
    "223001": "EVENSHROUD",
    "223002": "TRAILBLAZER",
    "223003": "ARCHANGELS_STAFF",
    "223004": "MANAMUNE",
    "223005": "GHOSTCRAWLERS",
    "223006": "BERSERKERS_GREAVES",
    "223009": "BOOTS_OF_SWIFTNESS",
    "223011": "CHEMTECH_PUTRIFIER",
    "223020": "SORCERERS_SHOES",
    "223026": "GUARDIAN_ANGEL",
    "223031": "INFINITY_EDGE",
    "223032": "YUN_TAL_WILDARROWS",
    "223033": "MORTAL_REMINDER",
    "223036": "LORD_DOMINIKS_REGARDS",
    "223039": "ATMAS_RECKONING",
    "223040": "SERAPHS_EMBRACE",
    "223042": "MURAMANA",
    "223046": "PHANTOM_DANCER",
    "223047": "PLATED_STEELCAPS",
    "223050": "ZEKES_CONVERGENCE",
    "223053": "STERAKS_GAGE",
    "223057": "SHEEN",
    "223065": "SPIRIT_VISAGE",
    "223067": "KINDLEGEM",
    "223068": "SUNFIRE_AEGIS",
    "223071": "BLACK_CLEAVER",
    "223072": "BLOODTHIRSTER",
    "223073": "EXPERIMENTAL_HEXPLATE",
    "223074": "RAVENOUS_HYDRA",
    "223075": "THORNMAIL",
    "223078": "TRINITY_FORCE",
    "223084": "HEARTSTEEL",
    "223085": "RUNAANS_HURRICANE",
    "223087": "STATIKK_SHIV",
    "223089": "RABADONS_DEATHCAP",
    "223091": "WITS_END",
    "223094": "RAPID_FIRECANNON",
    "223095": "STORMRAZOR",
    "223100": "LICH_BANE",
    "223102": "BANSHEES_VEIL",
    "223105": "AEGIS_OF_THE_LEGION",
    "223107": "REDEMPTION",
    "223109": "KNIGHTS_VOW",
    "223110": "FROZEN_HEART",
    "223111": "MERCURYS_TREADS",
    "223112": "GUARDIANS_ORB",
    "223115": "NASHORS_TOOTH",
    "223116": "RYLAIS_CRYSTAL_SCEPTER",
    "223118": "MALIGNANCE",
    "223119": "WINTERS_APPROACH",
    "223121": "FIMBULWINTER",
    "223124": "GUINSOOS_RAGEBLADE",
    "223135": "VOID_STAFF",
    "223137": "CRYPTBLOOM",
    "223139": "MERCURIAL_SCIMITAR",
    "223142": "YOUMUUS_GHOSTBLADE",
    "223143": "RANDUINS_OMEN",
    "223146": "HEXTECH_GUNBLADE",
    "223152": "HEXTECH_ROCKETBELT",
    "223153": "BLADE_OF_THE_RUINED_KING",
    "223156": "MAW_OF_MALMORTIUS",
    "223157": "ZHONYAS_HOURGLASS",
    "223158": "IONIAN_BOOTS_OF_LUCIDITY",
    "223161": "SPEAR_OF_SHOJIN",
    "223165": "MORELLONOMICON",
    "223172": "ZEPHYR",
    "223177": "GUARDIANS_BLADE",
    "223181": "HULLBREAKER",
    "223184": "GUARDIANS_HAMMER",
    "223185": "GUARDIANS_DIRK",
    "223190": "LOCKET_OF_THE_IRON_SOLARI",
    "223193": "GARGOYLE_STONEPLATE",
    "223222": "MIKAELS_BLESSING",
    "223302": "TERMINUS",
    "223504": "ARDENT_CENSER",
    "223508": "ESSENCE_REAVER",
    "223742": "DEAD_MANS_PLATE",
    "223748": "TITANIC_HYDRA",
    "223814": "EDGE_OF_NIGHT",
    "224004": "SPECTRAL_CUTLASS",
    "224005": "IMPERIAL_MANDATE",
    "224401": "FORCE_OF_NATURE",
    "224403": "THE_GOLDEN_SPATULA",
    "224628": "HORIZON_FOCUS",
    "224629": "COSMIC_DRIVE",
    "224633": "RIFTMAKER",
    "224636": "NIGHT_HARVESTER",
    "224637": "DEMONIC_EMBRACE",
    "224644": "CROWN_OF_THE_SHATTERED_QUEEN",
    "224645": "SHADOWFLAME",
    "224646": "STORMSURGE",
    "226035": "SILVERMERE_DAWN",
    "226333": "DEATHS_DANCE",
    "226609": "CHEMPUNK_CHAINSWORD",
    "226610": "SUNDERED_SKY",
    "226616": "STAFF_OF_FLOWING_WATER",
    "226617": "MOONSTONE_RENEWER",
    "226620": "ECHOES_OF_HELIA",
    "226621": "DAWNCORE",
    "226630": "GOREDRINKER",
    "226631": "STRIDEBREAKER",
    "226632": "DIVINE_SUNDERER",
    "226653": "LIANDRYS_ANGUISH",
    "226655": "LUDENS_COMPANION",
    "226656": "EVERFROST",
    "226657": "ROD_OF_AGES",
    "226662": "ICEBORN_GAUNTLET",
    "226664": "HOLLOW_RADIANCE",
    "226665": "JAKSHO,_THE_PROTEAN",
    "226667": "RADIANT_VIRTUE",
    "226671": "GALEFORCE",
    "226672": "KRAKEN_SLAYER",
    "226673": "IMMORTAL_SHIELDBOW",
    "226675": "NAVORI_FLICKERBLADES",
    "226676": "THE_COLLECTOR",
    "226691": "DUSKBLADE_OF_DRAKTHARR",
    "226692": "ECLIPSE",
    "226693": "PROWLERS_CLAW",
    "226694": "SERYLDAS_GRUDGE",
    "226695": "SERPENTS_FANG",
    "226696": "AXIOM_ARC",
    "226697": "HUBRIS",
    "226698": "PROFANE_HYDRA",
    "226699": "VOLTAIC_CYCLOSWORD",
    "226701": "OPPORTUNITY",
    "228001": "ANATHEMAS_CHAINS",
    "228002": "WOOGLETS_WITCHCAP",
    "228003": "DEATHBLADE",
    "228004": "ADAPTIVE_HELM",
    "228005": "OBSIDIAN_CLEAVER",
    "228006": "SANGUINE_BLADE",
    "228008": "RUNEGLAIVE",
    "228020": "ABYSSAL_MASK",
    "322065": "SHURELYAS_BATTLESONG",
    "323002": "TRAILBLAZER",
    "323003": "ARCHANGELS_STAFF",
    "323004": "MANAMUNE",
    "323040": "SERAPHS_EMBRACE",
    "323042": "MURAMANA",
    "323050": "ZEKES_CONVERGENCE",
    "323070": "TEAR_OF_THE_GODDESS",
    "323075": "THORNMAIL",
    "323107": "REDEMPTION",
    "323109": "KNIGHTS_VOW",
    "323110": "FROZEN_HEART",
    "323119": "WINTERS_APPROACH",
    "323121": "FIMBULWINTER",
    "323190": "LOCKET_OF_THE_IRON_SOLARI",
    "323222": "MIKAELS_BLESSING",
    "323504": "ARDENT_CENSER",
    "324005": "IMPERIAL_MANDATE",
    "326616": "STAFF_OF_FLOWING_WATER",
    "326617": "MOONSTONE_RENEWER",
    "326620": "ECHOES_OF_HELIA",
    "326621": "DAWNCORE",
    "326657": "ROD_OF_AGES",
    "328020": "ABYSSAL_MASK",
    "443054": "DARKSTEEL_TALONS",
    "443055": "FULMINATION",
    "443056": "DEMON_KINGS_CROWN",
    "443058": "SHIELD_OF_MOLTEN_STONE",
    "443059": "CLOAK_OF_STARRY_NIGHT",
    "443060": "SWORD_OF_THE_DIVINE",
    "443061": "FORCE_OF_ENTROPY",
    "443062": "SANGUINE_GIFT",
    "443063": "ELEISAS_MIRACLE",
    "443064": "TALISMAN_OF_ASCENSION",
    "443069": "HAMSTRINGER",
    "443079": "TURBO_CHEMTANK",
    "443080": "TWIN_MASK",
    "443081": "HEXBOLT_COMPANION",
    "443083": "WARMOGS_ARMOR",
    "443090": "REAPERS_TOLL",
    "443193": "GARGOYLE_STONEPLATE",
    "444636": "NIGHT_HARVESTER",
    "444637": "DEMONIC_EMBRACE",
    "444644": "CROWN_OF_THE_SHATTERED_QUEEN",
    "446632": "DIVINE_SUNDERER",
    "446656": "EVERFROST",
    "446667": "RADIANT_VIRTUE",
    "446671": "GALEFORCE",
    "446691": "DUSKBLADE_OF_DRAKTHARR",
    "446693": "PROWLERS_CLAW",
    "447100": "MIRAGE_BLADE",
    "447101": "GAMBLERS_BLADE",
    "447102": "REALITY_FRACTURE",
    "447103": "HEMOMANCERS_HELM",
    "447104": "INNERVATING_LOCKET",
    "447105": "EMPYREAN_PROMISE",
    "447106": "DRAGONHEART",
    "447107": "DECAPITATOR",
    "447108": "RUNECARVER",
    "447109": "CRUELTY",
    "447110": "MOONFLAIR_SPELLBLADE",
    "447111": "OVERLORDS_BLOODMAIL",
    "447112": "FLESHEATER",
    "447113": "DETONATION_ORB",
    "447114": "REVERBERATION",
    "447115": "REGICIDE",
    "447116": "KINKOU_JITTE",
    "447118": "PYROMANCERS_CLOAK",
    "447119": "LIGHTNING_ROD",
    "447120": "DIAMOND_TIPPED_SPEAR",
    "447121": "TWILIGHTS_EDGE",
    "447122": "BLACK_HOLE_GAUNTLET",
    "447123": "PUPPETEER",
}

def build_pid_role_map_by_team_position(match: Dict) -> Dict[int, str]:
    #Maps participantId to role like MIDDLE_B or JUNGLE_R using teamPosition. 
    pid_role_map = {}

    participants = match["metadata"]["info"].get("participants", [])
    for p in participants:
        pid = p.get("participantId")
        position = p.get("teamPosition", "UNKNOWN").upper()
        if not pid or position == "UNKNOWN":
            continue
        side = "B" if pid <= 5 else "R"
        pid_role_map[pid] = f"{position}_{side}"

    return pid_role_map


def replace_pid_with_roles(tokens: List[Any], pid_role_map: Dict[int, str]) -> List[str]:
    # Replaces [P1]...[P10] with their corresponding roles like [TOP_B].
    def flatten_tokens(tokens):
        for token in tokens:
            if isinstance(token, list):
                yield from flatten_tokens(token)
            else:
                yield token

    new_tokens = []
    for token in flatten_tokens(tokens):
        replaced_token = token
        for pid in range(1, 11):
            pid_tag = f"[P{pid}]"
            if pid_tag in replaced_token:
                replaced_token = replaced_token.replace(pid_tag, f"[{pid_role_map.get(pid, f'P{pid}')}]")
        new_tokens.append(replaced_token)
    return new_tokens

def get_item_category(item_id: str) -> str:
    # Get the category of an item based on its ID. 
    if item_id == 0:
        return "[EMPTY]"
    else :
        return ITEM_CATEGORY_MAP.get(str(item_id), f"[UNKNOWN_{item_id}]")

def process_skill_level_up(event: Dict[str, Any]) -> str:
    # Process skill level up events.
    participant_id = event["participantId"]
    skill_slot = event["skillSlot"]
    skill_name = SKILL_MAP.get(skill_slot, f"SLOT{skill_slot}")
    return f"[SKILL_UP][P{participant_id}][SKILL_{skill_name}]"

def process_item_transaction(event: Dict[str, Any]) -> str:
    # Process item purchase, sell, undo, and destroy events. 
    participant_id = event["participantId"]
    item_id = event.get("itemId")
    item_category = get_item_category(item_id)
    
    action_map = {
        "ITEM_PURCHASED": "BUY",
        "ITEM_SOLD": "SELL",
        "ITEM_DESTROYED": "USE"
    }
    
    action = action_map.get(event["type"], "UNKNOWN_ITEM_EVENT")
    return f"[ITEM_{action}][P{participant_id}][{item_category}]"


def process_item_undo(event: Dict[str, Any]) -> str:
    participant_id = event["participantId"]
    before_item =  get_item_category(event.get("beforeId"))
    after_item = get_item_category(event.get("afterId"))
    return f"[ITEM_UNDO][P{participant_id}][{before_item}][{after_item}]"
    

def process_ward_interaction(event: Dict[str, Any]) -> str:
    # Process ward placement and kill events.
    participant_id = event.get("participantId") or event.get("killerId")
    ward_type = WARD_TYPE_MAP.get(event.get("wardType", "UNKNOWN_WARD_T"), "UNKNOWN_WARD")

    action = "PLACE" if event["type"] == "WARD_PLACED" else "KILL"
    return f"[WARD_{action}][P{participant_id}][{ward_type}]"

def process_combat(event: Dict[str, Any]) -> List[str]:
    # Process champion kill events.
    tokens = []
    killer_id = event["killerId"]
    victim_id = event.get("victimId")
    assists = event.get("assistingParticipantIds", [])
    bounty = event.get("bounty", 0)
    
    tokens.append(
        f"[KILL][P{killer_id}][P{victim_id}]"
    )
    
    # Format assists
    if assists:
        tokens.append("[ASSIST]")
        for aid in assists:
            tokens.append(f"[P{aid}]")
    
    tokens.append(f"[BOUNTY][{bounty}]")
    return tokens


def process_building(event: Dict[str, Any]) -> str:
    # Process building kill events with strategic weights.
    tokens = []
    building_type = event.get("buildingType", "").replace("_BUILDING", "")
    killer_id = event.get("killerId")
    assists = event.get("assistingParticipantIds", [])
    
    tokens.append(f"[BUILDING_DESTROY][{building_type}]")
    
    if building_type == "TOWER":
        tower_type = event.get("towerType", "UNKNOWN_TOWER").replace("_TURRET", "")
        tokens.append(f"[{tower_type}]")
        
    tokens.append(f"[P{killer_id}]")
    
    if assists:
        tokens.append("[ASSIST]")
        for aid in assists:
            tokens.append(f"[P{aid}]")
  
    return tokens

def process_position(event: Dict[str, Any]) -> str:
    # Process position events. 
    position = event.get("position", {})
    x, y = position.get("x", 0), position.get("y", 0)
    
    # Determine location type based on coordinates
    location = "UNKNOWN_LOCATION"
    
    # Bot lane
    if x < 6000 and y < 6000:
        location = "BOT_LANE"
    # Mid lane - diagonal range
    elif abs(x - y) < 1000:  # Within ~1000 units of diagonal
        location = "MID_LANE"
    # Dragon pit
    elif 9300 <= x <= 10300 and 3900 <= y <= 4900:
        location = "DRAGON_PIT"
    # Baron pit
    elif 4500 <= x <= 5500 and 9900 <= y <= 10900:
        location = "BARON_PIT"
    # Top lane can be inferred from high y values
    elif y > 8000:
        location = "TOP_LANE"
    else:
        location = "JUNGLE"  # Default to jungle for other areas
    
    return f"[POS][{location}][{x},{y}]"

def process_game_state(event: Dict[str, Any]) -> str:
    # Process game state events.
    if event["type"] == "GAME_END":
        winning_team = event.get("winningTeam", 0)
        return f"[GAME_END][TEAM{winning_team}]"
    return ""

def process_special_kill(event: Dict[str, Any]) -> str:
    # Process special kill events (multi-kills, first blood, etc.). 
    killer_id = event["killerId"]
    kill_type = event.get("killType", "NORMAL")
    if kill_type == "KILL_MULTI":
        streak = event["multiKillLength"]
        if streak == 2:
            kill_type = "DOUBLE_KILL"
        elif streak == 3:
            kill_type = "TRIPLE_KILL"
        elif streak == 4:
            kill_type = "QUADRA_KILL"
        elif streak == 5:
            kill_type = "PENTA_KILL"
    
    return f"[SPECIAL_{kill_type}][P{killer_id}]"

def process_elite_monster(event: Dict[str, Any]) -> str:
    # Process elite monster kill events with strategic weights.
    tokens = []
    monster_type = event.get("monsterType", "UNKNOWN_MON")
    killer_id = event.get("killerId")
    assists = event.get("assistingParticipantIds", [])
    
    tokens.append(f"[MONSTER_{monster_type}]")
    
    if monster_type == "DRAGON":
        monster_subtype = event.get("monsterSubType", "UNKNOWN_MON_S2")
        tokens.append(f"[{monster_subtype}]")
        
    tokens.append(f"[P{killer_id}]")
    
    if assists:
        tokens.append("[ASSIST]")
        for aid in assists:
            tokens.append(f"[P{aid}]")
    return tokens

def process_turret_plate(event: Dict[str, Any]) -> str:
    # Process turret plate destruction events.
    killer_id = event.get("killerId")
    lane_type = event.get("laneType", "UNKNOWN_LANE")
    return f"[BUILDING_PLATE][{lane_type}][P{killer_id}]"

def process_level_up(event: Dict[str, Any]) -> str:
    # Process level up events.
    participant_id = event["participantId"]
    level = event.get("level", 0)
    return f"[LEVEL_UP][P{participant_id}][{level}]"

def normalize_rank(full_rank: str) -> str:
    # Extracts the rank tier from a full rank like 'DIAMOND I' or 'GOLD IV'.
    return full_rank.split()[0] if full_rank and isinstance(full_rank, str) else "UNRANKED"

def tokenize_match(timeline_json: Dict[str, Any], tokens: List[str]) -> List[str]:
    tokens.append("[GAME_START]")
    last_kill_time = 0

    for frame in timeline_json["info"]["frames"]:
        tokens.append("[FRAME]")
        for event in frame.get("events", []):
            event_type = event.get("type")
            timestamp = event.get("timestamp", 0)

            if event_type == "SKILL_LEVEL_UP":
                tokens.append(process_skill_level_up(event))
            elif event_type == "ITEM_UNDO":
                tokens.append(process_item_undo(event))
            elif event_type in ["ITEM_PURCHASED", "ITEM_SOLD", "ITEM_DESTROYED"]:
                tokens.append(process_item_transaction(event))
            elif event_type in ["WARD_PLACED", "WARD_KILL"]:
                tokens.append(process_ward_interaction(event))
            elif event_type == "CHAMPION_KILL":
                tokens.extend(process_combat(event))  # already returns a list
                last_kill_time = timestamp
            elif event_type == "CHAMPION_SPECIAL_KILL":
                if timestamp - last_kill_time < 1000:
                    tokens.append(process_special_kill(event))
            elif event_type == "ELITE_MONSTER_KILL":
                tokens.extend(process_elite_monster(event))  # already returns a list
            elif event_type == "BUILDING_KILL":
                tokens.extend(process_building(event))  # already returns a list
            elif event_type == "TURRET_PLATE_DESTROYED":
                tokens.append(process_turret_plate(event))
            elif event_type == "LEVEL_UP":
                tokens.append(process_level_up(event))
            elif event_type == "GAME_END":
                tokens.append(process_game_state(event))

    return tokens

def process_match_file(file_path: str) -> List[str]:
    # Process a match file and return the token sequence.
    with open(file_path, 'r') as f:
        match_data = json.load(f)
        rank = normalize_rank(match_data["rank"])
        tokens = []
        tokens.append(f"[RANK_{rank}]")
    return tokenize_match(match_data["timeline"], tokens)

def save_tokens_for_training(tokens_by_match: List[List[str]], output_file: str):
    # Save tokens in a format suitable for training, with each match's tokens on a new line.
    
    def flatten(match_tokens):
        return list(chain.from_iterable(
            t if isinstance(t, list) else [t] for t in match_tokens
        ))
    
    with open(output_file, 'w') as f:
        for match_tokens in tokens_by_match:
            flat_tokens = flatten(match_tokens)
            # Join tokens for this match with spaces and write to file
            f.write(" ".join(flat_tokens) + "\n")

if __name__ == "__main__":

    # Enable for batched processing
    # all_matches = []
    # for filename in json_files:
    #     print(f"Processing {filename}...")
    #     with open(filename, 'r') as f:
    #         matches = json.load(f)
    #         all_matches.extend(matches)

    with open(MATCH_FILE, 'r') as f:
        matches = json.load(f)
    # Keep tokens separated by match
    all_match_tokens = []
    for match in matches:
        rank = normalize_rank(match["rank"])
        tokens = [f"[RANK_{rank}]"]
        match_tokens = tokenize_match(match["timeline"], tokens)

        pid_role_map = build_pid_role_map_by_team_position(match)
        match_tokens = replace_pid_with_roles(match_tokens, pid_role_map)

        all_match_tokens.append(match_tokens)
        print(f"Processed match {match['match_id']} - {len(match_tokens)} tokens")
    
    # Save tokens in training format
    save_tokens_for_training(all_match_tokens, OUTPUT_FILE)
    
    # Print summary
    print(f"\nProcessed {len(all_match_tokens)} matches")
    print(f"Average tokens per match: {sum(len(tokens) for tokens in all_match_tokens) / len(all_match_tokens):.2f}")
    print(f"Total tokens: {sum(len(tokens) for tokens in all_match_tokens)}") 