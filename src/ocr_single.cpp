/* ocr_single.cpp
 * 
 * This file is part of OCR.
 * 
 * Copyright 2012 David B. Knoester.
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <assert.h>
#include <cmath>
#include <algorithm>
#include <ea/evolutionary_algorithm.h>
#include <ea/generational_models/synchronous.h>
#include <ea/generational_models/death_birth_process.h>
#include <ea/representations/circular_genome.h>
#include <ea/fitness_function.h>
#include <ea/cmdline_interface.h>
#include <ea/datafiles/generation_fitness.h>
#include <fn/hmm/hmm_network.h>
#include <fn/hmm/hmm_evolution.h>
using namespace ea;

#include "ocr_game.h"
#include "ocr_statistics.h"

/*! Fitness function for the OCR problem.
 */
struct ocr_fitness : fitness_function<unary_fitness<double>, constantS, absoluteS, stochasticS> {
    games::ocr_game game;

    template <typename EA>
    void initialize(EA& ea) {
        fn::hmm::options::NODE_INPUT_FLOOR = get<HMM_INPUT_FLOOR>(ea);
		fn::hmm::options::NODE_INPUT_LIMIT = get<HMM_INPUT_LIMIT>(ea);
		fn::hmm::options::NODE_OUTPUT_FLOOR = get<HMM_OUTPUT_FLOOR>(ea);
		fn::hmm::options::NODE_OUTPUT_LIMIT = get<HMM_OUTPUT_LIMIT>(ea);        
        
        game.initialize(get<GAME_OCR_LABELS>(ea), get<GAME_OCR_IMAGES>(ea), get<GAME_OUTPUT_WIDTH>(ea));
        check_argument(game.num_inputs()==get<HMM_INPUT_N>(ea), "game and HMM input numbers differ");
        check_argument(game.num_outputs()==get<HMM_OUTPUT_N>(ea), "game and HMM output numbers differ");
    }

    template <typename RNG>
    games::ocr_game::results game_results(fn::hmm::hmm_network& network, std::size_t game_size, std::size_t updates, RNG& rng) {
        return game.play(network, game_size, updates, rng);
    }
    
	template <typename Individual, typename RNG, typename EA>
	double operator()(Individual& ind, RNG& rng, EA& ea) {
		fn::hmm::hmm_network network(ind.repr(), get<HMM_INPUT_N>(ea), get<HMM_OUTPUT_N>(ea), get<HMM_HIDDEN_N>(ea));
        
        games::ocr_game::results r = game_results(network, get<GAME_SIZE>(ea), get<HMM_UPDATE_N>(ea), rng);
        
        put<FF_RNG_SEED>(get<FF_RNG_SEED>(ea), ind);
        put<OCR_TPR>(r.mean_tpr(), ind);
        put<OCR_TNR>(r.mean_tnr(), ind);
        put<OCR_FPR>(r.mean_fpr(), ind);
        put<OCR_FNR>(r.mean_fnr(), ind);
        put<OCR_OUT>(r.unique_outputs(), ind);
        put<OCR_ACC>(r.mean_accuracy(), ind);
        put<OCR_ORDER>((r.mean_tpr()+r.mean_tnr()-r.mean_fpr()-r.mean_fnr()) / (r.mean_tpr()+r.mean_tnr()+r.mean_fpr()+r.mean_fnr()), ind);
        put<OCR_IMAGES>(algorithm::vcat(r.idx.begin(), r.idx.end()), ind);

        return 1.0 + get<OCR_ORDER>(ind);
        
//        typedef std::vector<double> distance_vector;
//        distance_vector dv;
//        for(std::size_t i=0; i<10; ++i) {
//            dv.push_back(r.tpr(i) * r.tnr(i) * (1.0-r.fpr(i)) * (1.0-r.fnr(i)));
//        }
//        
//        return 1.0 + ea::algorithm::vmag(dv.begin(), dv.end());
    }
};



//! Evolutionary algorithm definition.
typedef evolutionary_algorithm<
circular_genome<unsigned int>,
hmm_mutation,
ocr_fitness,
recombination::asexual,
generational_models::death_birth_process,
initialization::complete_population<hmm_random_individual>
> ea_type;


/*! Define the EA's command-line interface.
 */
template <typename EA>
class ocr : public cmdline_interface<EA> {
public:
    virtual void gather_options() {
        // hmm options
        add_option<HMM_INPUT_N>(this);
        add_option<HMM_OUTPUT_N>(this);
        add_option<HMM_HIDDEN_N>(this);
        add_option<HMM_UPDATE_N>(this);
        add_option<HMM_INPUT_FLOOR>(this);
        add_option<HMM_INPUT_LIMIT>(this);
        add_option<HMM_OUTPUT_FLOOR>(this);
        add_option<HMM_OUTPUT_LIMIT>(this);
        
        // game options
        add_option<GAME_SIZE>(this);
        add_option<GAME_OCR_LABELS>(this);
        add_option<GAME_OCR_IMAGES>(this);
        add_option<GAME_OUTPUT_WIDTH>(this);
        
        // ea options
        add_option<REPRESENTATION_SIZE>(this);
        add_option<POPULATION_SIZE>(this);
        add_option<REPLACEMENT_RATE_P>(this);
        add_option<MUTATION_GENOMIC_P>(this);
        add_option<MUTATION_PER_SITE_P>(this);
        add_option<MUTATION_UNIFORM_INT_MAX>(this);
        add_option<MUTATION_DELETION_P>(this);
        add_option<MUTATION_DUPLICATION_P>(this);
        add_option<TOURNAMENT_SELECTION_N>(this);
        add_option<TOURNAMENT_SELECTION_K>(this);
        add_option<RUN_UPDATES>(this);
        add_option<RUN_EPOCHS>(this);
        add_option<CHECKPOINT_PREFIX>(this);
        add_option<RNG_SEED>(this);
        add_option<RECORDING_PERIOD>(this);
        
        // analysis options
        add_option<ANALYSIS_INPUT>(this);
        add_option<ANALYSIS_OUTPUT>(this);
        add_option<ANALYSIS_ROUNDS>(this);
    }
    
    virtual void gather_tools() {
        add_tool<hmm_genetic_graph>(this);
        add_tool<hmm_reduced_graph>(this);
        add_tool<hmm_detailed_graph>(this);
        add_tool<hmm_causal_graph>(this);
    }
    
    virtual void gather_events(EA& ea) {
        add_event<datafiles::generation_fitness>(this, ea);
        add_event<mean_roc_trajectory>(this, ea);
    };
};
LIBEA_CMDLINE_INSTANCE(ea_type, ocr);
