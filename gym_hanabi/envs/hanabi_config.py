import collections

Config = collections.namedtuple('Config', [
    "colors",                    # string list
    "max_tokens",                # int,
    "max_fuses",                 # int,
    "hand_size",                 # int,
    "card_counts",               # int list,
    "num_turns_after_last_deal", # int,
    "num_players",               # int,
])

HANABI_CONFIG = Config(
    colors=["red", "green", "blue", "yellow", "white"],
    max_tokens=8,
    max_fuses=4,
    hand_size=5,
    card_counts=[3, 2, 2, 2, 1],
    num_turns_after_last_deal=2,
    num_players=2,
)

MEDIUM_HANABI_CONFIG = Config(
    colors=["red", "green", "blue", "yellow"],
    max_tokens=8,
    max_fuses=4,
    hand_size=4,
    card_counts=[3, 2, 2, 1],
    num_turns_after_last_deal=2,
    num_players=2,
)

MINI_HANABI_CONFIG = Config(
    colors=["red", "green", "blue"],
    max_tokens=6,
    max_fuses=3,
    hand_size=3,
    card_counts=[2, 2, 1],
    num_turns_after_last_deal=2,
    num_players=2,
)

MINI_HANABI_4P_CONFIG = Config(
    colors=["red", "green", "blue"],
    max_tokens=6,
    max_fuses=3,
    hand_size=3,
    card_counts=[2, 2, 1],
    num_turns_after_last_deal=4,
    num_players=4,
)

MINI_HANABI_LOTSOFINFO_CONFIG = \
    MINI_HANABI_CONFIG._replace(max_tokens=30)

MINI_HANABI_LOTSOFTURNS_CONFIG = \
    MINI_HANABI_CONFIG._replace(num_turns_after_last_deal=15)
