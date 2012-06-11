#ifndef _OCR_STATISTICS_H_
#define _OCR_STATISTICS_H_

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/max.hpp>

/*! Datafile for mean generation, and mean & max fitness.
 */
template <typename EA>
struct mean_roc_trajectory : record_statistics_event<EA> {
    mean_roc_trajectory(EA& ea) : record_statistics_event<EA>(ea), _df("mean_roc_trajectory.dat") {
        _df.add_field("update")
        .add_field("mean_tpr", "mean true positive rate")
        .add_field("mean_fpr", "mean false positive rate")
        .add_field("mean_acc", "mean accuracy")
        .add_field("mean_order", "mean order param, (tp+tn-fp-fn) / (tp+tn+fp+fn)")
        .add_field("dom_order", "dominate order param");
    }
    
    virtual ~mean_roc_trajectory() {
    }
    
    virtual void operator()(EA& ea) {
        using namespace boost::accumulators;
        accumulator_set<double, stats<tag::mean,tag::max> > tpr, fpr, acc, order;
        
        
        for(typename EA::population_type::iterator i=ea.population().begin(); i!=ea.population().end(); ++i) {
            tpr(get<OCR_TPR>(ind(i,ea)));
            fpr(get<OCR_FPR>(ind(i,ea)));
            acc(get<OCR_ACC>(ind(i,ea)));
            double o=(get<OCR_TPR>(ind(i,ea)) + get<OCR_TNR>(ind(i,ea)) - get<OCR_FPR>(ind(i,ea)) - get<OCR_FNR>(ind(i,ea))) 
            / (get<OCR_TPR>(ind(i,ea)) + get<OCR_TNR>(ind(i,ea)) + get<OCR_FPR>(ind(i,ea)) + get<OCR_FNR>(ind(i,ea)));
            order(o);
        }
        _df.write(ea.current_update())
        .write(mean(tpr))
        .write(mean(fpr))
        .write(mean(acc))
        .write(mean(order))
        .write(max(order))
        .endl();
    }
    
    datafile _df;
};


template <typename EA>
struct roc_trajectory : record_statistics_event<EA> {
    roc_trajectory(EA& ea) : record_statistics_event<EA>(ea), _df("roc_trajectory.dat") {
        _df.add_field("update")
        .add_field("mean_tpr", "mean true positive rate")
        .add_field("mean_fpr", "mean false positive rate")
        .add_field("mean_acc", "mean accuracy")
        .add_field("dom_tpr", "dominant individual true positive rate")
        .add_field("dom_fpr", "dominant individual false positive rate")
        .add_field("dom_acc", "dominant individual accuracy");
    }
    
    virtual ~roc_trajectory() {
    }
    
    virtual void operator()(EA& ea) {
        using namespace boost::accumulators;
        accumulator_set<double, stats<tag::mean> > tpr, fpr, acc;
        
        for(typename EA::population_type::iterator i=ea.population().begin(); i!=ea.population().end(); ++i) {
            tpr(get<OCR_TPR>(ind(i,ea)));
            fpr(get<OCR_FPR>(ind(i,ea)));
            acc(get<OCR_ACC>(ind(i,ea)));
        }
        
        typename EA::individual_type& indi = analysis::find_most_fit_individual(ea);
        
        _df.write(ea.current_update())
        .write(mean(tpr))
        .write(mean(fpr))
        .write(mean(acc))
        .write(get<OCR_TPR>(indi))
        .write(get<OCR_FPR>(indi))
        .write(get<OCR_ACC>(indi))
        .endl();
    }
    
    datafile _df;
};



#endif