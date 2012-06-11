#ifndef _FN_HMM_GPU_NETWORK_H_
#define _FN_HMM_GPU_NETWORK_H_

#include <limits>

#include <fn/hmm/hmm_network.h>
#include <fn/hmm/gpu_update.h>

namespace fn {
	namespace hmm {
		
		/*! GPU-enabled Hidden Markov Model Network.
		 */
		class gpu_network : public hmm_network {
		public:
			//! Constructor.
			gpu_network(const genome& g, std::size_t in, std::size_t out, std::size_t hidden);
			
			//! Destructor.
			virtual ~gpu_network();
			
			//! Update this network.
			template <typename InputIterator, typename OutputIterator, typename RNG>
			void update(InputIterator first, InputIterator last, OutputIterator result, RNG& rng) {
				_h->rotate();
				std::copy(first, last, _h->tminus1_begin());
				gpu_update(_h, _device_mem, rng(std::numeric_limits<int>::max()));
				std::copy(_h->t_output_begin(), _h->t_output_end(), result);
			}	
			
		protected:
			void* _device_mem; //!< Device memory.
		};
		
	} // hmm
} // fn

#endif
