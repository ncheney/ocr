#include <arpa/inet.h>
#include <boost/scoped_array.hpp>
#include <fstream>
#include <set>
#include <ea/algorithm.h>
#include <ea/exceptions.h>
#include "ocr_game.h"


/*! Initialize this game.
 */
void games::ocr_game::initialize(const std::string& lname, const std::string& iname, unsigned int width) {
    using namespace std;
    
    _width = width;

    // Read in the labels:
    //
    ifstream ifs(lname.c_str(), ios::binary);
    if(!ifs.good()) {
        throw ea::file_io_exception("could not open: " + lname + " for reading");
    }
    
    // check that the magic number is right:
    unsigned int magic=0;
    ifs.read((char*)&magic, sizeof(magic));
    magic = ntohl(magic); // convert from file to host byte order
    assert(magic == 2049);
    
    // check that the file has more than 0 records:
    unsigned int lrecords=0;
    ifs.read((char*)&lrecords, sizeof(lrecords));
    lrecords = ntohl(lrecords);
    assert(lrecords > 0);
    
    // read in all the labels in one fell swoop:
    boost::scoped_array<unsigned char> labels(new unsigned char[lrecords]);
    ifs.read(reinterpret_cast<char*>(labels.get()), lrecords);
    ifs.close();

    
    // Read in the images:
    //
    ifs.open(iname.c_str(), ios::binary);
    if(!ifs.good()) {
        throw ea::file_io_exception("could not open: " + iname + " for reading");
    }
    
    // check that the magic number is right:
    magic=0;
    ifs.read((char*)&magic, sizeof(magic));
    magic = ntohl(magic); // convert from file to host byte order
    assert(magic == 2051);
    
    // check that the file has more than 0 records:
    unsigned int irecords=0;
    ifs.read((char*)&irecords, sizeof(irecords));
    irecords = ntohl(irecords);
    assert(irecords > 0);

    // sanity; make sure that our labels & images have the same number of records
    assert(irecords == lrecords);
    
    // read in the size of the images:
    unsigned int rows, cols;
    ifs.read((char*)&rows, sizeof(rows));
    rows = ntohl(rows);
    ifs.read((char*)&cols, sizeof(cols));
    cols = ntohl(cols);
    
    // read in the images, match them up with their labels, and figure out 
    // how many labels we have (we need this to determine the number of outputs):
    std::set<char> lset;
    boost::scoped_array<unsigned char> img(new unsigned char[rows*cols]);

    for(unsigned int i=0; i<irecords; ++i) {
        lset.insert(labels[i]); // incrementally build up our label set...
        ifs.read(reinterpret_cast<char*>(img.get()), rows*cols); // read the image
        if(!ifs.good()) {
            throw ea::file_io_exception("could not read from: " + lname);
        }

        // now, build the labeled_image struct:
        _idb.push_back(labeled_image(labels[i], img.get(), rows*cols));
    }
    assert(_idb.size() == irecords);
    
    // and figure out how many inputs and outputs the network needs:
    _nin = rows*cols;
    _nout = lset.size() * _width;
}

