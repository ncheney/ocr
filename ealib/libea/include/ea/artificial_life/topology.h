#ifndef _EA_TOPOLOGY_H_
#define _EA_TOPOLOGY_H_

#include <boost/iterator/iterator_facade.hpp>
#include <boost/serialization/nvp.hpp>
#include <utility>
#include <vector>
#include <ea/meta_data.h>

namespace ea {
    
    /*! Well-mixed topology.
     */
    template <typename EA>
    struct well_mixed {
        typedef EA ea_type; //<! EA type using this topology.        
        typedef typename ea_type::individual_ptr_type individual_ptr_type; //!< Pointer to individual type.
        
        //! Location type.
        struct location_type {
            std::size_t idx; //!< Location index.
            individual_ptr_type p; //!< Individual (if any) at this location.
        };
        
        /*! Orientation type.
         (Null, as there are no orientations in a well-mixed environment.)
         */
        struct orientation_type { };
        
        typedef std::vector<location_type> location_list_type; //!< Container type for locations.

        /*! Well-mixed neighborhood iterator.
         
         The idea here is that the underlying topology of a well-mixed environment is random.  This
         iterator class thus provides a random (w/ replacement) sequence of locations.
         
         The "end" iterator is really just a a number of dereferences, set to the number of possible
         locations.  Note that this is *not* the same as iterating over all locations.
         */
        struct iterator : boost::iterator_facade<iterator, location_type, boost::single_pass_traversal_tag> {
            //! Constructor.
            iterator(std::size_t n, location_list_type& locs, ea_type& ea) : _n(n), _locs(locs), _ea(ea) {
            }
            
            //! Increment operator.
            void increment() { ++_n; }
            
            //! Iterator equality comparison.
            bool equal(iterator const& that) const { return _n == that._n; }
            
            //! Dereference this iterator.
            location_type& dereference() const { return *_ea.rng().choice(_locs.begin(), _locs.end()); }
            
            std::size_t _n; //!< how many times this iterator has been dereferenced.
            location_list_type& _locs; //!< list of all possible locations.
            ea_type& _ea; //!< EA (used for rngs, primarily).
        };                
        
        //! Initialize this topology.
        void initialize(ea_type& ea) {
            _locs.resize(get<POPULATION_SIZE>(ea));
        }
        
        //!< Retrieve the neighborhood of the given individual.
        std::pair<iterator,iterator> neighborhood(individual_ptr_type p, ea_type& ea) {
            return std::make_pair(iterator(0,_locs,ea), iterator(_locs.size(),_locs,ea));
        }                
        
        //! Replace the organism (if any) living in location l with p.
        template <typename AL>
        void replace(location_type& l, individual_ptr_type p, AL& al) {
            // kill the occupant of l, if any
            if(l.p) {
                l.p->alive() = false;
            }
            l.p = p;
        }
        
        //! Serialize this topology.
        template <class Archive>
        void serialize(Archive& ar, const unsigned int version) {
        }
        
        location_list_type _locs; //!< List of all locations in this topology.
    };
    
} // ea

#endif
