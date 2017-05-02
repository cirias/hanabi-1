from gym_hanabi.envs import hanabi

class HeuristicPolicy(object):
    def __init__(self, env):
        self.env = env
        self.config = self.env.config
        self.reward = self.env.reward
        self.spaces = self.env.spaces

        err = "HeuristicPolicy only supports 2 players."
        assert self.config.num_players == 2, err

    def compute_play_or_discard(self, all_info, played_cards):
        """
        Return cards that we think are playable, and the candidate for a
        discard. A card is playable if we have number information for it that
        matches a card that could be played next. The candidate to discard is a
        card whose duplicate has already been played. If we can't find such a
        card, then it is a card for which we have the least amount of
        information.

        TODO: Determine situations when we should play a card with only color
              information.
        TODO: Discard cards in FIFO order, if there are multiple with the least
              amount of information.
        """
        play_cards = []
        discard_card = None
        for card, info in enumerate(all_info):
            color, number = info

            if number is None:
                # Don't try to play a card with no number information. If that
                # color pile is already complete, discard it.
                num_numbers = len(self.config.card_counts)
                if color is not None and played_cards[color] == num_numbers:
                    discard_card = card
                continue

            if color is None:
                # If all color piles are already past this number, discard the
                # card.
                all_colors_played = True
                for color in self.config.colors:
                    if played_cards[color] < number:
                        all_colors_played = False
                if all_colors_played:
                    discard_card = card

                # Check if the card's number is playable with respect to any
                # one of the colors.
                match = False
                for color in self.config.colors:
                    if number == played_cards[color] + 1:
                        match = True
                if match:
                    # If the check succeeds, add this card to the cards we
                    # think are playable.
                    play_cards.append(card)
            elif number == played_cards[color] + 1:
                # If we have color information, check if the card's number is
                # playable. If the check succeeds, add this card to the to top
                # of the cards we think are playable.
                play_cards = [card] + play_cards
            elif number <= played_cards[color]:
                # If this color pile is already past this number, discard the
                # card.
                discard_card = card

        # If we weren't able to find a card that we could definitely discard,
        # then discard one with the least amount of information.
        less_info = lambda lhs, rhs: not (lhs is not None and rhs is None)
        if discard_card is None:
            discard_card = 0
            for card, info in enumerate(all_info):
                color, number = info
                if (less_info(color, all_info[discard_card].color) and
                    less_info(number, all_info[discard_card].number)):
                    # This card has the same or less information than the
                    # current card we want to discard.
                    discard_card = card

        return play_cards, discard_card


    def compute_information(self, observation):
        """
        Return information moves that we think will be helpful, based on the
        cards that we think the other player will play or discard using
        compute_play_or_discard. This is added to in order:

        1) Color information about the cards that the other will play that will
           result in a fuse.
        2) Number information about the card that the other will discard, if
           the card is eventually playable and has no duplicates left.
        3) Number information about the cards that the other has color
           information about, if the card is playable.
        4) Information about the cards that the other has no information about,
           if the card is playable. If the other player has other cards with
           the same number and a different color, and no color information
           for those cards, then give color information for the card we
           want them to play.  Else, give number information.
        """
        them = observation.players[0]
        them.cards = [card for card in them.cards if card is not None]
        them.info = [info for info in them.info if info is not None]

        information = []
        their_cards_to_play, their_card_to_discard = self.compute_play_or_discard(
                them.info, observation.played_cards)
        # (1)
        # NOTE: Color information is sufficient to prevent the other player from
        # playing this card, since compute_play_or_discard only returns cards to
        # play if we have number information.
        for card in their_cards_to_play:
            color, number = them.cards[card]
            if number != observation.played_cards[color] + 1:
                information.append(hanabi.InformColorMove(color, 0))
        if information:
            return information

        # (2)
        card = them.cards[their_card_to_discard]
        if card.number > observation.played_cards[card.color]:
            discard_count = observation.discarded_cards.count(card)
            if discard_count + 1 == self.config.card_counts[card.number - 1]:
                information.append(hanabi.InformNumberMove(card.number, 0))
        if information:
            return information

        # (3)
        color_info_cards = [card_index for card_index, info in
            enumerate(them.info) if info.color is not None and
            info.number is None]
        for card_index in color_info_cards:
            card = them.cards[card_index]
            if card.number == observation.played_cards[card.color] + 1:
                information.append(hanabi.InformNumberMove(card.number, 0))
        if information:
            return information

        # (4)
        no_info_cards = [card_index for card_index, info in
                enumerate(them.info) if info.color is None and
                info.number is None]
        for card_index in no_info_cards:
            card = them.cards[card_index]
            if card.number == observation.played_cards[card.color] + 1:
                # A duplicate has the same number but a different color.
                duplicates = [i for i, dup in enumerate(them.cards) if
                        dup.number == card.number and dup.color != card.color]
                # Only give color information about the duplicate if they don't
                # already have information about its color.
                duplicates = [i for i in duplicates if
                        them.info[i].color is None]
                if duplicates:
                    information.append(hanabi.InformColorMove(card.color, 0))
                else:
                    information.append(hanabi.InformNumberMove(card.number, 0))
        return information

    def get_move(self, observation_sample):
        observation = self.spaces.sample_to_observation(observation_sample)

        play_cards, discard_card = self.compute_play_or_discard(
            observation.your_info, observation.played_cards)

        # Pretend to apply the moves that we would play, so we don't inform the
        # other player about a card that we might already have.
        for card in play_cards:
            number_info, color_info = observation.your_info[card]
            if color_info is not None:
                # If we have color information about the card we'll play, that
                # color pile will definitely increase.
                observation.played_cards[color_info] += 1
            else:
                # If we don't have color information about the card that we'll
                # play, a color pile whose next number matches may increase.
                # Apply the hypothetical move if exactly one color pile
                # matches.
                possible_colors = [color for color, played_number in
                        observation.played_cards.items() if number_info ==
                        played_number + 1]
                if len(possible_colors) == 1:
                    observation.played_cards[possible_colors[0]] += 1

        if observation.num_tokens > 0:
            information = self.compute_information(observation)
        else:
            information = []

        if information:
            return information[0]
        elif play_cards:
            return hanabi.PlayMove(play_cards[0])
        else:
            return hanabi.DiscardMove(discard_card)

    def get_action(self, observation):
        return (self.spaces.action_to_sample(self.get_move(observation)), )

