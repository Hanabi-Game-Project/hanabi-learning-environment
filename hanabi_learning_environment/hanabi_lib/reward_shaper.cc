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

// hanabi_learning_env::RewardShaper::RewardShaper(
//     double performance = 0.,
//     double m_play_penalty = 0.,
//     double m_play_reward = 0.,
//     int num_ranks = -1,
//     hanabi_learning_env::RewardShapingParams params = hanabi_learning_env::RewardShapingParams (true, 0.8, 0., 0., 0., 0., 0.),
//     hanabi_learning_env::ShapingType shape_type = hanabi_learning_env::ShapingType(0, 1, 2, 3)
// ){

//     this->performance = performance;
//     this->m_play_penalty = m_play_penalty;
//     this->m_play_reward = m_play_reward;
//     this->num_ranks = num_ranks;
//     this->shape_type = shape_type;
//     this->unshaped = std::make_tuple(0., this->shape_type.NONE);
//     this->params = params;
// }

// double hanabi_learning_env::RewardShaper::GetPerformance(){
//     return this -> performance ; 
// }


void hanabi_learning_env::RewardShaper::Performance(double performance) {
    
    this -> performance = performance ; 
    this -> m_play_penalty = this -> params.m_play_penalty * this -> performance ;
    this -> m_play_reward = this -> params.m_play_reward * this -> performance ;
}


std::tuple<std::vector<double>, std::vector<int>> hanabi_learning_env::RewardShaper::Shape(
    std::vector<hanabi_learning_env::HanabiObservation> observations, 
    std::vector<hanabi_learning_env::HanabiMove> moves
){
    
    assert(observations.size() == moves.size()) ; 

    if ( this -> num_ranks == -1){
        for ( int i=0; i<observations.size(); i++){
            this -> num_ranks = observations[i].ParentGame()-> NumRanks() ; 
        }
    }

    std::vector<std::tuple<double, int>> shaped_rewards ; 

    for(int i=0 ; i<observations.size(); i++ ){
        shaped_rewards.push_back(hanabi_learning_env::RewardShaper::Calculate(observations[i], moves[i])) ; 
    }


    std::vector<double> rewards ;
    std::vector <int> shapings ; 

    for(int i=0; i<shaped_rewards.size(); i++){
        rewards.push_back(std::get<0>(shaped_rewards[i]))  ;
        shapings.push_back(std::get<1>(shaped_rewards[i])) ;  
    }

    return std::make_tuple(rewards, shapings) ; 
}

std::tuple<double, int> hanabi_learning_env::RewardShaper::Calculate(
    hanabi_learning_env::HanabiObservation observation,
    hanabi_learning_env::HanabiMove move){

    if (move.MoveType() == hanabi_learning_env::HanabiMove::kPlay){
        return hanabi_learning_env::RewardShaper::PlayShape(observation, move) ; 
    }
    else if (move.MoveType() == hanabi_learning_env::HanabiMove::kDiscard){
        return hanabi_learning_env::RewardShaper::DiscardShape(observation, move) ; 
    }
    else if (move.MoveType() == hanabi_learning_env::HanabiMove::kRevealColor 
                || move.MoveType() == hanabi_learning_env::HanabiMove::kRevealRank){
        return hanabi_learning_env::RewardShaper::HintShape(observation, move) ; 
    }
    else {
        return this -> unshaped ; 
    }
}

std::tuple<double, int> hanabi_learning_env::RewardShaper::HintShape(
    hanabi_learning_env::HanabiObservation observation,
    hanabi_learning_env::HanabiMove move){
    
    return this -> unshaped ;
}




std::tuple<double, int> hanabi_learning_env::RewardShaper::PlayShape(
    hanabi_learning_env::HanabiObservation observation,
    hanabi_learning_env::HanabiMove move
    ) {

    double prob ; 

    // If card has already been played -> Then not playable 
    // If the cards below the rank of same color are in discard pile, we can't play it 

    
    if (! observation.CardPlayableOnFireworks(move.Color(), move.Rank()) ){
        // return this->unshaped ; 

        this -> DiscardShape(observation, move) ;

    }
    else{
        try {
            prob = observation.PlayablePercent()[move.CardIndex()];
        }
        catch(const std::out_of_range& oor){
            std::cerr << "Out of Range error: " << oor.what() << '\n';
            return this -> unshaped ; 
        }

        if (prob < this -> params.min_play_probability ){
            double penalty = this -> params.w_play_penalty + this -> m_play_penalty ; 
            return std::make_tuple(penalty, this -> shape_type.RISKY) ;  
        }

        double reward = this -> params.w_play_reward + this -> m_play_reward ;
        return std::make_tuple(reward, this -> shape_type.CONSERVATIVE) ;
    }


    
}


std::tuple<double, int> hanabi_learning_env::RewardShaper::DiscardShape(
    hanabi_learning_env::HanabiObservation observation,
    hanabi_learning_env::HanabiMove move){

    std::vector<hanabi_learning_env::HanabiCard> discard_pile = observation.DiscardPile();
    int8_t card_index = move.CardIndex();
    hanabi_learning_env::HanabiCard discarded_Card = observation.GetCardToDiscard(card_index);

    if (this -> params.penalty_last_of_kind == 0){
        return this -> unshaped ;  
    }
    else if (discard_pile.size() == 0 ){
        return this -> unshaped ; 
    }

    int count_card = 0;
    int max_score = observation.get_max_score(discarded_Card.Color()) ;  

    for(int i=0; i<discard_pile.size(); i++){
        if (discarded_Card.Rank() == discard_pile[i].Rank() 
                && discarded_Card.Color() == discard_pile[i].Color()){
                    count_card ++ ; 
        }

    }

    if ( count_card 
            == observation.ParentGameRef().NumberCardInstances(discarded_Card.Color(), discarded_Card.Rank()) - 1 ){

        if ( discarded_Card.Rank() < max_score){
            return  std::make_tuple(
                max_score - discarded_Card.Rank() , 
                this -> shape_type.DISCARD_LAST_OF_KIND 
            ) ; 
        }
    }



    // // If the card is a 5 in rank, then  it is indispensible
    // if (discarded_Card.Rank() == (this ->  num_ranks - 1) ) {
    //     return std::make_tuple(this -> params.penalty_last_of_kind, shape_type.DISCARD_LAST_OF_KIND) ; 
    // }
    // // No need for shaping if the discard pile is not present
    // else if (discard_pile.size() == 0 ){
    //     return this -> unshaped ; 
    // }
    // // Else if rank is not 1, then if another card with the same rank and color exists, 
    // // then shape the reward
    // else if (discarded_Card.Rank() > 0 ){

    //     for (int i=0; i < discard_pile.size(); i++){
    //         if( discarded_Card.Rank() == discard_pile[i].Rank() 
    //                 && discarded_Card.Color() == discard_pile[i].Color() ){
    //             return std::make_tuple( this -> params.penalty_last_of_kind, 
    //                                 shape_type.DISCARD_LAST_OF_KIND) ; 
    //         }

    //         return this -> unshaped ; 
    //     }
    // }
    // //For all other cases, see if there are 1s of the discarded card's color already in teh pile, in which 
    // // case shape the reward with a penalty
    // else {
    //     int counter = 0 ;

    //     for (int i = 0; i < discard_pile.size(); i++){
    //         if ( discard_pile[i].Rank() == 0 && discard_pile[i].Color() == discarded_Card.Color() ){
    //             counter += 1 ; 
    //         }
    //     }

    //     if ( counter = 2 ){
    //         return std::make_tuple( this -> params.penalty_last_of_kind, shape_type.DISCARD_LAST_OF_KIND) ; 
    //     }
    //     else {
    //         return this -> unshaped ; 
    //     }
    // }
};
