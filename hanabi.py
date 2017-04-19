import random
from collections import namedtuple
from collections import defaultdict


MAX_FUSES = 4
MAX_TOKENS = 8
COLORS = ["WHITE", "YELLOW", "GREEN", "BLUE", "RED"]
CARD_COUNTS = [3, 2, 2, 2, 1]
INITIAL_HAND = 5


Card = namedtuple('Card', ['color', 'number', 'color_information', 'number_information'])
ColorInformation = namedtuple('ColorInformation', ['player', 'color'])
NumberInformation = namedtuple('NumberInformation', ['player', 'number'])
DiscardMove = namedtuple('DiscardMove', ['index'])
PlayMove = namedtuple('PlayMove', ['index'])


class IllegalMoveException(Exception):
    pass


class PlayerState():
    def __init__(self, player_index):
        self.player_index = player_index
        self.hand = []

    def add_card(self, card):
        self.hand.append(card)

    def remove_card(self, card_index):
        if card_index < 0 or card_index >= len(self.hand):
            raise IllegalMoveException("Tried to remove illegal card {} from "
                                       "player {}".format(card_index,
                                                          self.player_index))
        return self.hand.pop(card_index)

    def check_number(self, index, number):
        return self.hand[index][1] == number

    def check_color(self, index, color):
        return self.hand[index][0] == color

    def to_string(self):
        cards = ["{}:{}".format(card.color, card.number) for card in self.hand]
        return ' '.join(cards)


class GameState():
    def __init__(self, num_players):
        self.num_tokens = MAX_TOKENS
        self.fuses = MAX_FUSES
        self.played_cards = defaultdict(list)
        self.discarded = []
        self.turn = 0
        # Initialize the deck.
        self.deck = []
        for color in COLORS:
            for number, count in enumerate(CARD_COUNTS):
                self.deck += [Card(color, number + 1, None, None)] * count
        # Initialize the players with 5 random cards each.
        self.players = []
        for i in range(num_players):
            self.players.append(PlayerState(i))
        for i in range(INITIAL_HAND):
            for player_index in range(num_players):
                self.deal_card(player_index)

    def deal_card(self, player_index):
        if len(self.deck) == 0:
            raise LastRoundException("Out of cards to deal")
        player = self.players[player_index]
        card_index = random.randint(0, len(self.deck) - 1)
        card = self.deck.pop(card_index)
        player.add_card(card)

    def get_num_rounds_left(self):
        if len(self.played_cards) == len(COLORS):
            for i in range(len(COLORS)):
                if len(self.played_cards[i]) == len(CARD_COUNTS):
                    return 0
        if len(self.deck) == 0:
            return 1
        if self.fuses == 0:
            return 0
        return -1

    def get_current_view(self):
        view = []
        for player_index, player in enumerate(self.players):
            if player_index == self.turn:
                card_view = [(card.color_information, card.number_information) for card in player.hand]
            else:
                card_view = player.hand
            view.append((player_index, card_view))
        return view

    def play(self, moves):
        last_round_countdown = -1
        for move in moves: 
            try:
                self.apply_move(move)
            except IllegalMoveException as e:
                print e.message
                continue

            self.turn += 1
            self.turn %= len(self.players)

            num_rounds = self.get_num_rounds_left()
            if num_rounds == 0:
                break
            elif num_rounds == 1:
                last_round_countdown = len(self.players) + 1
            last_round_countdown -= 1
            if last_round_countdown == 0:
                break

    def apply_information_move(self, move, is_color):
        if self.num_tokens == 0:
            raise IllegalMoveException("No more information tokens left")
        if move.player == self.turn:
            raise IllegalMoveException("Player {} cannot give information to "
                                       "himself".format(move.player))

        # Save the information in the player's hand.
        player = self.players[move.player]
        for i, card in enumerate(player.hand):
            if is_color:
                if card.color != move.color:
                    continue
                card = Card(card.color, card.number, move_information, card.number_information)
            else:
                if card.number != move.number:
                    continue
                card = Card(card.color, card.number, card.color, card.number_information)
            player.hand[i] = card

        # Giving information uses one token.
        self.num_tokens -= 1

    def apply_move(self, move):
        if isinstance(move, ColorInformation):
            self.apply_information_move(move, True)
        elif isinstance(move, NumberInformation):
            self.apply_information_move(move, False)
        elif isinstance(move, DiscardMove):
            player = self.players[self.turn]
            # Move the card to the discarded pile.
            card = player.remove_card(move.index)
            self.discarded.append(card)
            # Get one information token back.
            self.num_tokens += 1
            # Replace the card.
            self.deal_card(self.turn)
        elif isinstance(move, PlayMove):
            player = self.players[self.turn]
            # Try to move the card to the played cards. If it's impossible, one
            # fuse gets used up.
            card = player.remove_card(move.index)
            stack = self.played_cards[card.color]
            if card.number == len(stack) + 1:
                stack.append(card)
            else:
                self.fuses -= 1
            # Replace the card.
            self.deal_card(self.turn)
        else:
            raise IllegalMoveException("Unexpected move type")
