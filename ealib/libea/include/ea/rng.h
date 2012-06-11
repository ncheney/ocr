#ifndef _EA_RNG_H_
#define _EA_RNG_H_

#include <boost/random.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/nvp.hpp>
#include <iterator>
#include <vector>
#include <limits>
#include <sstream>
#include <ctime>
#include <ea/algorithm.h>


namespace ea {
	
	/*! Provides useful abstractions for dealing with random numbers.
	 
	 Note: when many random numbers are needed, consider using the uniform_X_rng
	 methods.  These methods return a generator that can be quickly queried for new 
	 random numbers, as opposed to the uniform_X methods, that build a new generator 
	 and use it once for each call.
	 */
	template <typename Engine>
	class rng {
	public:
		//! Underlying source of randomness.
		typedef Engine engine_type;
		//! Uniform real distribution.
		typedef boost::uniform_real< > uniform_real_dist;
		//! Uniform integer distribution.
		typedef boost::uniform_int< > uniform_int_dist;
		//! Generator for uniformly-distributed random real numbers.
		typedef boost::variate_generator<engine_type&, uniform_real_dist> real_rng_type;
		//! Generator for uniformly-distributed random integers.
		typedef boost::variate_generator<engine_type&, uniform_int_dist> int_rng_type;
		//! Normal real distribution.
		typedef boost::normal_distribution< > normal_real_dist;
		//! Generator for normally-distributed random real numbers.
		typedef boost::variate_generator<engine_type&, normal_real_dist> normal_real_rng_type;
		//! Normal integer distribution.
		typedef boost::normal_distribution<int> normal_int_dist;
		//! Generator for normally-distributed random integers.
		typedef boost::variate_generator<engine_type&, normal_int_dist> normal_int_rng_type;
		//! Type of the argument to this RNG (to support concept requirements).
		typedef int argument_type;
		//! Type returned by this RNG (to support concept requirements).
		typedef int result_type;
		
		//! Constructor.
		rng() {
			reset(static_cast<unsigned int>(std::time(0)));
		}
		
		//! Constructor with specified rng seed.
		rng(unsigned int s) {
			reset(s);
		}
		
		//! Reset this random number generator with the specified seed.
		void reset(unsigned int s) {
			if(s == 0) {
				s = static_cast<unsigned int>(std::time(0));
			}
			_eng.seed(s);
			_p.reset(new real_rng_type(_eng, uniform_real_dist(0.0,1.0)));
			_bit.reset(new int_rng_type(_eng, uniform_int_dist(0,1)));			
		}
		
		/*! Returns a random number in the range [0,n).
		 
		 This method enables this class to be used as a RandomNumberGenerator in
         STL algorithms.
		 */
		result_type operator()(argument_type n) {
			return uniform_integer_rng(0,n)();
		}
		
		/*! Test a probability.
		 
		 Returns true if P < prob, false if P >= prob.  Prob must be in the range [0,1].
		 */
		bool p(double prob) { assert((prob >= 0.0) && (prob <= 1.0)); return (*_p)() < prob; }

		//! Returns a random bit.
		bool bit() { return (*_bit)(); }
		
		//! Returns a random real value uniformly drawn from the range [min, max) 
		double uniform_real(double min, double max) { return uniform_real_rng(min,max)(); }

		//! Returns a random real value uniformly drawn from the range (min, max).
		double uniform_real_nz(double min, double max) {
            double r = uniform_real(min,max);
            while(r == 0.0) {
                r = uniform_real(min,max);
            }
            return r;
        }

		//! Returns a random number generator of reals over the range [min, max).
		real_rng_type uniform_real_rng(double min, double max) {
			return real_rng_type(_eng, uniform_real_dist(min,max));
		}

		//! Returns a random real value drawn from a normal distribution with the given mean and variance.
		double normal_real(double mean, double variance) {
			return normal_real_rng_type(_eng, normal_real_dist(mean, variance))();
		}
		
		/*! Returns an integer value in the range [min, max).
		 
		 For consistency with most other random number generators, max will never be
		 returned.
		 */
		int uniform_integer(int min, int max) { return uniform_integer_rng(min,max)(); }
		
        /*! Returns a random integer.
         */
        int uniform_integer() { return uniform_integer(std::numeric_limits<int>::min(), std::numeric_limits<int>::max()); }
        
		/*! Returns a random number generator of integers over the range [min, max).

		 For consistency with most other random number generators, max will never be
		 returned.  (Boost's integer rng allows max to be returned, thus the -1 below.)
		 */
		int_rng_type uniform_integer_rng(int min, int max) {
			return int_rng_type(_eng, uniform_int_dist(min,max-1));
		}

        /*! Generates random numbers into the given output iterator.
         */
        template <typename T, typename OutputIterator>
        void generate(std::size_t n, T min, T max, OutputIterator oi) {
            typedef std::set<int> set_type;
            set_type v;
            while(v.size() < n) {
                int i = uniform_integer(min, max);
                if(v.find(i) == v.end()) {
                    v.insert(i);
                    *oi++ = i;
                }
            }
        }
        
		/*! Returns a normally-distributed integer with the given mean and variance.
		 */
		int normal_int(int mean, int variance) {
			return static_cast<int>(normal_real(static_cast<double>(mean), static_cast<double>(variance))+0.5);
		}
		
		/*! Choose two different random numbers from [min,max), and return them in sorted order.
		 */
		template <typename T>
		std::pair<T,T> choose_two(T min, T max) {
			int_rng_type irng = uniform_integer_rng(static_cast<int>(min), static_cast<int>(max));
			int one=irng();	int two=irng();
			while(one == two) {
				two = irng();
			}
			if(one > two) { std::swap(one, two); }
			return std::make_pair(static_cast<T>(one), static_cast<T>(two));
		}
		
		/*! Choose two different iterators from the range [f,l), and return them in sorted order (r.first occurs before r.second).
		 */
		template <typename ForwardIterator>
		std::pair<ForwardIterator,ForwardIterator> choose_two_range(ForwardIterator f, ForwardIterator l) {
			int_rng_type irng = uniform_integer_rng(0, static_cast<int>(std::distance(f,l)));
			int one=irng();	int two=irng();
			while(one == two) {
				two = irng();
			}

			return std::make_pair(f+one, f+two);
		}

		/*! Sample elements uniformly with replacement from the given range, copying them to the output range.
		 */
		template <typename InputIterator, typename OutputIterator>
		void sample_with_replacement(InputIterator first, InputIterator last, OutputIterator output, std::size_t n) {
			std::size_t range = std::distance(first, last);
			int_rng_type irng = uniform_integer_rng(0,range);
			for( ; n>0; --n) {
                InputIterator t=first;
                std::advance(t,irng());
				*output++ = *t;
			}
		}

		/*! Sample n elements uniformly without replacement from [f,l), copying them to output.
		 */
		template <typename InputIterator, typename OutputIterator>
		void sample_without_replacement(InputIterator first, InputIterator last, OutputIterator output, std::size_t n) {
			typedef std::vector<std::size_t> replacement_type;
			std::size_t range = std::distance(first, last);
			replacement_type replace(range);
			algorithm::iota(replace.begin(), replace.end());
			
			for(; n>0; --n) {
				replacement_type::iterator i=replace.begin();
				std::advance(i, uniform_integer(0, replace.size()));
                InputIterator t=first;
                std::advance(t,*i);
				*output++ = *t;
				replace.erase(i);
			}
		}
		
		/*! Returns a randomly-selected iterator from the given range, selected with replacement.
		 */
		template <typename InputIterator>
		InputIterator choice(InputIterator first, InputIterator last) {
			std::advance(first, uniform_integer(0, std::distance(first, last)));
            return first;
		}
		
		/*! Returns a randomly-selected iterator from the given range, selected without replacement.
		 
		 The passed-in replacement map is used to ensure that no single element from [first,last) is
		 returned more than once.  If an empty replacement map is passed-in, it will be initialized,
		 as an empty map would indicate that the entire range was selected.
		 
		 //! Replacement map, used to track elements that were selected for a range for without-replacment methods.
		 typedef std::vector<std::size_t> replacement_map_type;
		 */
		template <typename InputIterator, typename ReplacementMap>
		InputIterator choice(InputIterator first, InputIterator last, ReplacementMap& rm) {
			if(rm.empty()) {
				rm.resize(std::distance(first,last));
				iota(rm.begin(), rm.end(), 0);
			}
			
			typename ReplacementMap::iterator i=rm.begin();
			std::advance(i, uniform_integer(0, rm.size()));
			std::advance(first, *i);
			rm.erase(i);
			return first;
		}

	private:
		//! Disallowing copy construction.
		rng(const rng&);
		//! Disallowing assignment operator.
		rng* operator=(const rng&);
		
		engine_type _eng; //!< Underlying generator of randomness.
		boost::scoped_ptr<real_rng_type> _p; //!< Generator for probabilities.
		boost::scoped_ptr<int_rng_type> _bit; //!< Generator for bits.

		// These enable serialization and de-serialization of the rng state.
		friend class boost::serialization::access;
		template<class Archive>
		void save(Archive & ar, const unsigned int version) const {
			std::ostringstream out;
			out << _eng;
			std::string state(out.str());
			ar & BOOST_SERIALIZATION_NVP(state);
		}
		
		template<class Archive>
		void load(Archive & ar, const unsigned int version) {
			std::string state;
			ar & BOOST_SERIALIZATION_NVP(state);
			std::istringstream in(state);
			in >> _eng;
		}
		BOOST_SERIALIZATION_SPLIT_MEMBER();
	};
	
	//! Default random number generation type.
	typedef rng<boost::mt19937> default_rng_type;

} // util

#endif
