// Copyright 2020 Aditya Mohan (adityak735@gmail.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include <functional>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>
#include <cassert>

#include "hanabi_game.h"
#include "hanabi_move.h"
#include "hanabi_observation.h"
#include "hanabi_state.h"
#include "observation_encoder.h"
#include "util.h"
#include "hanabi_parallel_env.h"
#include "reward_shaper.h"


namespace hanabi_learning_env {

void RewardShaper::Performance(double performance) {
    
    this -> performance = performance ; 
    this -> m_play_penalty = this -> params.m_play_penalty * this -> performance ;
    this -> m_play_reward = this -> params.m_play_reward * this -> performance ;
}


std::tuple<std::vector<double>, std::vector<RewardShaper::Type>>
hanabi_learning_env::RewardShaper::Shape(
    std::vector<HanabiObservation> observations,
    std::vector<HanabiMove> moves
){
    
    assert(observations.size() == moves.size()) ; 

    std::vector<double> rewards(observations.size(), 0.0) ;
    std::vector <RewardShaper::Type> shapings(observations.size(), RewardShaper::Type::kNone) ;

    for(int i=0; i<observations.size(); i++){
    	std::tuple<double, RewardShaper::Type> shaped_rewards =
    			RewardShaper::Calculate(observations[i], moves[i]);

    	rewards[i] = std::get<0>(shaped_rewards);
    	shapings[i] = std::get<1>(shaped_rewards);

    }

    return std::make_tuple(rewards, shapings) ; 
}


std::tuple<double, RewardShaper::Type>
RewardShaper::Calculate(HanabiObservation observation, HanabiMove move){

    if (move.MoveType() == HanabiMove::kPlay){
        return RewardShaper::PlayShape(observation, move) ;
    }
    else if (move.MoveType() == HanabiMove::kDiscard){
        return RewardShaper::DiscardShape(observation, move) ;
    }
    else if (move.MoveType() == HanabiMove::kRevealColor
                || move.MoveType() == HanabiMove::kRevealRank){
        return RewardShaper::HintShape(observation, move) ;
    }
    else {
        return this -> unshaped ; 
    }
}


std::tuple<double, RewardShaper::Type>
RewardShaper::HintShape(HanabiObservation observation, HanabiMove move){
    return this -> unshaped ;
}


std::tuple<double, RewardShaper::Type>
RewardShaper::PlayShape(HanabiObservation observation, HanabiMove move) {

    double prob ;
    double add_reward = 0;
    RewardShaper::Type type = RewardShaper::Type::kNone;

	// playability shaping
	try {

		// get the playability of the card to be played
		prob = observation.PlayablePercent()[move.CardIndex()];

		// depending on playability value: define penalty or reward for playing the card
		if (prob < this -> params.min_play_probability ){
			add_reward = this -> params.w_play_penalty + this -> m_play_penalty;
			type = RewardShaper::Type::kRisky;
		} else {
			add_reward = this -> params.w_play_reward + this -> m_play_reward;
			type = RewardShaper::Type::kConservative;
		}

	} catch(const std::out_of_range& oor){
		std::cerr << "Out of Range error: " << oor.what() << '\n';
	}

	// get the card to be played and determine if it is playable on fireworks
	// if card is not playable, also do discard shaping as card will be added to discard pile
	// add additional discard shaping penalty and overwrite shaping type
	if (! observation.CardPlayableOnFireworks(observation.GetCardToDiscard(move.CardIndex())) ) {

		std::tuple<double, RewardShaper::Type> discard_shape =
				this-> DiscardShape(observation, move);

		if (std::get<1>(discard_shape) != RewardShaper::Type::kNone){
			add_reward += std::get<0>(discard_shape) ;
			type = std::get<1>(discard_shape) ;
		}
	}

	return std::make_tuple(add_reward, type);

}


std::tuple<double, RewardShaper::Type>
RewardShaper::DiscardShape(HanabiObservation observation, HanabiMove move){

    std::vector<HanabiCard> discard_pile = observation.DiscardPile();
    int8_t card_index = move.CardIndex();
    HanabiCard discarded_Card = observation.GetCardToDiscard(card_index);

//    if (this -> params.penalty_last_of_kind == 0){
//        return this -> unshaped ;
//    }
//    else if (discard_pile.size() == 0 ){
    if (discard_pile.size() == 0 ){
        return this -> unshaped ; 
    }

    int count_card = 0;
    int max_score = observation.ParentState()->CalculateMaxScore(discarded_Card.Color()) ; 
    
    for(int i=0; i<discard_pile.size(); i++){
        if (discarded_Card == discard_pile[i]){
        	count_card ++ ;
        }
    }

    if ( count_card 
            == observation.ParentGameRef().NumberCardInstances(discarded_Card) - 1 ){

        if ( discarded_Card.Rank() < max_score){
            return  std::make_tuple(
                (max_score - discarded_Card.Rank()) * this -> params.penalty_last_of_kind ,
				RewardShaper::Type::kDiscardLastOfKind
            ) ; 
        }
    }

    return this -> unshaped ;
}

} //end namespace hanabi_learning_env
