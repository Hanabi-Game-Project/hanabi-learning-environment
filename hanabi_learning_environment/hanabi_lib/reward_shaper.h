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

#ifndef __REWARD_SHAPER_H__
#define __REWARD_SHAPER_H__

#include <functional>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <map>
#include <tuple>
#include <vector>

#include "hanabi_game.h"
#include "hanabi_move.h"
#include "hanabi_observation.h"
#include "hanabi_state.h"
#include "observation_encoder.h"
#include "util.h"
#include "hanabi_parallel_env.h"

namespace hanabi_learning_env {

    struct RewardShapingParams
    {

        RewardShapingParams(
            bool SHAPER = true,
            double MIN_PLAY_PROBABILITY = 0.8,
            double W_PLAY_PENALTY = 0.,
            double M_PLAY_PENALTY = 0.,
            double W_PLAY_REWARD = 0.,
            double M_PLAY_REWARD = 0.,
            double PENALTY_LAST_OF_KIND = 0.) : shaper(SHAPER),
                                                min_play_probability(MIN_PLAY_PROBABILITY),
                                                w_play_penalty(W_PLAY_PENALTY),
                                                m_play_penalty(M_PLAY_PENALTY),
                                                w_play_reward(W_PLAY_REWARD),
                                                m_play_reward(M_PLAY_REWARD),
                                                penalty_last_of_kind(PENALTY_LAST_OF_KIND) {}

        bool shaper;
        double min_play_probability;
        double w_play_penalty;
        double m_play_penalty;
        double w_play_reward;
        double m_play_reward;
        double penalty_last_of_kind;
    };

    class RewardShaper{

        public:

    	enum Type { kNone, kRisky, kDiscardLastOfKind, kConservative};

        /** \brief Constructor to assign default values
         *  \param performance Performance parameter to calculate the reward and penalties
         *  \param m_play_penalty Penalty for each play 
         *  \param m_play_reward  Reward for each play 
         *  \param num_ranks      Number of Ranks in the game
         *  \param params         Object having the information pertaining to the parameters to be used 
         *  \param shape_type     Scheme to be used for shaping
         */ 
        RewardShaper(
            RewardShapingParams params = hanabi_learning_env::RewardShapingParams()
        ) {
            this->performance = 0;
            this->m_play_penalty = 0;
            this->m_play_reward = 0;
            this->unshaped = std::make_tuple(0., Type::kNone);
            this->params = params;
        }

        /** \brief Return the performance parameter
         */
        double GetPerformance()   
        { return this->performance; }

        /** \brief Set the rewards and penalties based on Performance
         *  \param performance custom performance value to be used in setting te parameters
         */ 
        void Performance(double performance) ;


        /** \brief Calculate the shape of the rewards based on a set of moves and observations
         *  \param observation Vector of Observations
         *  \param move        The current move that will determine the shaping
         */
        std::tuple<std::vector<double>, std::vector<Type>> Shape (
            std::vector<HanabiObservation> observations, 
            std::vector<HanabiMove> moves
            ) ; 

        /** \brief Calculate the shape of the reward based on the move and observation
         *  \param observation Observation based on the move
         *  \param move        The current move that will determine the shaping
         */ 
        std::tuple<double, Type> Calculate(HanabiObservation observation, HanabiMove move) ;

        /** \brief Shape the reward for a hint move
         *  \param observation Observation based on the move
         *  \param move        The current move that will determine the shaping
         */
        std::tuple<double, Type> HintShape(HanabiObservation observation, HanabiMove move);

        /** \brief Shape the reward for a play move
         *  \param observation Observation based on the move
         *  \param move        The current move that will determine the shaping
         */
        std::tuple<double, Type> PlayShape(HanabiObservation observation, HanabiMove move) ;

        /** \brief Shape the reward for a discard move
         *  \param observation Observation based on the move
         *  \param move        The current move that will determine the shaping
         */
        std::tuple<double, Type> DiscardShape ( HanabiObservation observation, HanabiMove move) ;

        private:
        
			double performance;
			double m_play_penalty;
			double m_play_reward;
			RewardShapingParams params;
			std::tuple<double, Type> unshaped;
    };

}

#endif
