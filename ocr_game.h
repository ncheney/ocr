#ifndef _OCR_GAME_H_
#define _OCR_GAME_H_

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/shared_array.hpp>
#include <algorithm>
#include <iterator>
#include <functional>
#include <numeric>
#include <string>
#include <vector>
#include <set>
#include <string.h>
#include <fn/hmm/hmm_network.h>
#include <ea/meta_data.h>
#include <ea/generators.h>
#include <ea/algorithm.h>

LIBEA_MD_DECL(OCR_TPR, "individual.ocr.mean_tpr", double);
LIBEA_MD_DECL(OCR_TNR, "individual.ocr.mean_tnr", double);
LIBEA_MD_DECL(OCR_FPR, "individual.ocr.mean_fpr", double);
LIBEA_MD_DECL(OCR_FNR, "individual.ocr.mean_fnr", double);
LIBEA_MD_DECL(OCR_OUT, "individual.ocr.unique_outputs", double);
LIBEA_MD_DECL(OCR_ACC, "individual.ocr.mean_accuracy", double);
LIBEA_MD_DECL(OCR_IMAGES, "individual.ocr.images", std::string);

LIBEA_MD_DECL(GAME_SIZE, "game.ocr.size", int);
LIBEA_MD_DECL(GAME_OCR_LABELS, "game.ocr.label_filename", std::string);
LIBEA_MD_DECL(GAME_OCR_IMAGES, "game.ocr.image_filename", std::string);
LIBEA_MD_DECL(GAME_OUTPUT_WIDTH, "game.ocr.output_width", unsigned int);

namespace games {
    
	/*! OCR game.
     */
	class ocr_game {
	public:
		//! Struct that contains information about a single image.
		struct labeled_image {
			typedef std::vector<unsigned char> image_type; //!< "image" type
			
			//! Constructor.
			labeled_image(unsigned char l, unsigned char* f, std::size_t n) : label(l) {
				img.insert(img.end(), f, f+n);
                std::transform(img.begin(), img.end(), img.begin(), std::bind2nd(std::not_equal_to<unsigned char>(), 0));
			}
            
			unsigned char label; //!< label for this image
			image_type img; //!< image
		};

        //! Results of playing the OCR game.
        struct results {
            typedef std::vector<std::size_t> index_vector; //!< Type for a list of indices into the image db.
            enum field { P=0, N, TP, FP, TN, FN, LAST }; //!< Indices of positives, negatives, true positives, and false positives in the ROC table.

            //! Constructor.
            template <typename Generator>
            results(std::size_t n, Generator g) {
                memset(roc, 0, sizeof(roc));  
                std::generate_n(std::back_inserter(idx), n, g);
            }

            double mean_tpr() {
                double x=0.0;
                for(int i=0; i<10; ++i) {
                    if(roc[i][P] > 0) {
                        x += static_cast<double>(roc[i][TP]) / static_cast<double>(roc[i][P]);
                    }
                }
                return x / 10.0;
            }

            double mean_tnr() {
                double x=0.0;
                for(int i=0; i<10; ++i) {
                    if(roc[i][N] > 0) {
                        x += static_cast<double>(roc[i][TN]) / static_cast<double>(roc[i][N]);
                    }
                }
                return x / 10.0;
            }
            
            double mean_fpr() {
                double x=0.0;
                for(int i=0; i<10; ++i) {
                    if(roc[i][N] > 0) {
                        x += static_cast<double>(roc[i][FP]) / static_cast<double>(roc[i][N]);
                    }
                }
                return x / 10.0;
            }
            
            double mean_fnr() {
                double x=0.0;
                for(int i=0; i<10; ++i) {
                    if(roc[i][P] > 0) {
                        x += static_cast<double>(roc[i][FN]) / static_cast<double>(roc[i][P]);
                    }
                }
                return x / 10.0;
            }
            
            double tpr(std::size_t i) {
                if(roc[i][P] == 0) {
                    return 0.0;
                }
                return static_cast<double>(roc[i][TP]) / static_cast<double>(roc[i][P]);
            }

            double tnr(std::size_t i) {
                if(roc[i][N] == 0) {
                    return 0.0;
                }
                return static_cast<double>(roc[i][TN]) / static_cast<double>(roc[i][N]);
            }
            
            double fpr(std::size_t i) {
                if(roc[i][P] == 0) {
                    return 0.0;
                }
                return static_cast<double>(roc[i][FP]) / static_cast<double>(roc[i][P]);
            }
            
            double fnr(std::size_t i) {
                if(roc[i][N] == 0) {
                    return 0.0;
                }
                return static_cast<double>(roc[i][FN]) / static_cast<double>(roc[i][N]);
            }

            double unique_outputs() {
                double x=0.0;
                for(int i=0; i<10; ++i) {
                    if(roc[i][TP] || roc[i][FP]) {
                        ++x;
                    }
                }
                return x;
            }
            
            double accuracy(std::size_t i) {
                if(roc[i][P] + roc[i][N] > 0) {
                    return static_cast<double>(roc[i][TP] + roc[i][TN]) / static_cast<double>(roc[i][P] + roc[i][N]);
                } else {
                    return 0.0;
                }
            }
            
            double mean_accuracy() {
                double acc=0.0;
                double n=0.0;
                for(std::size_t i=0; i<10; ++i) {
                    if(roc[i][P] + roc[i][N] > 0) {
                        acc += (static_cast<double>(roc[i][TP] + roc[i][TN])) / (static_cast<double>(roc[i][P] + roc[i][N]));
                        ++n;
                    }
                }
                return acc / n;
            }
            
            index_vector idx; //!< Indices of the images that were tested
            int roc[10][LAST]; //!< label x [P, N, TP, FP]
        };
		
		typedef std::vector<labeled_image> imagedb_type; //!< Type for a list of labeled images.
		typedef std::vector<int> feature_vector; //!< Feature fector type; input & output from the HMM.
        
		//! Constructor.
		ocr_game() : _nin(0), _nout(0) {
		}
        
        //! Initialize this game.
		void initialize(const std::string& lname, const std::string& iname, unsigned int width);

		//! Return the number of features used for input.
		unsigned int num_inputs() {
			return _nin;
		}
		
		//! Return the number of features used for output.
		unsigned int num_outputs() {
			return _nout;
		}
		
		//! Play the game.
		template <typename RNG>
		results play(fn::hmm::hmm_network& network, std::size_t game_size, std::size_t updates, RNG& rng) {
            //results r(game_size, rng.uniform_integer_rng(0, _idb.size())); // results from the game
            results r(game_size, ea::series_generator<std::size_t>(0,1));

            for(results::index_vector::iterator i=r.idx.begin(); i!=r.idx.end(); ++i) {
                labeled_image& li=_idb[*i]; // the image we're testing
                feature_vector inputs(li.img.begin(), li.img.end()); // inputs to the HMM
                feature_vector outputs; // outputs from the HMM
                
                network.update_n(updates, inputs.begin(), inputs.end(), std::back_inserter(outputs), rng);

                // oh, sweet sanity!
                assert(outputs.size() == num_outputs());
                assert(num_outputs() == (10*_width));
                
                // track roc info (j is label, k is output bit)
                for(std::size_t j=0,k=0; k<outputs.size(); ++j,k+=_width) {
                    int on = ea::algorithm::vxor(&outputs[k], &outputs[k+_width]);
                    
                    if(li.label == j) {
                        ++r.roc[j][results::P]; // positives
                        if(on) {
                            ++r.roc[j][results::TP]; // true positives
                        } else {
                            ++r.roc[j][results::FN]; // false negative
                        }
                    } else {
                        ++r.roc[j][results::N]; // negatives
                        if(on) {
                            ++r.roc[j][results::FP]; // false positives
                        } else {
                            ++r.roc[j][results::TN]; // true negative
                        }
                    }
                }
            }
            return r;
        }
		
	protected:
        unsigned int _width; //!< width of output labels
		unsigned int _nin; //!< number of inputs
		unsigned int _nout; //!< number of outputs
		imagedb_type _idb; //!< image database
	};
	
} // games

#endif
