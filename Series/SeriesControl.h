#ifndef SERIESCONTROL_H
#define SERIESCONTROL_H

#include "SeriesClass.h"
#include "../Records/RecordClass.h"
#include <vector>
#include <string>

class SeriesControl {
private:
    SeriesClass series;
    std::string simplifyCommodityName(const std::string& fullName) const;
    std::string sanitizeFilename(const std::string& filename) const;
    std::string getColorByIndex(size_t index) const;
    void saveDataToFile(const SeriesClass& series, const std::string& filename) const;
    void saveGrowthCurveData(const std::vector<double>& values, 
                           const std::vector<std::string>& timestamps,
                           const std::string& filename, 
                           const std::string& name) const;
    void saveGrowthCurveData(const SeriesClass& series, 
                           const std::string& filename, 
                           const std::string& name) const;
    void cleanupDataFiles(const std::vector<std::string>& filenames) const;
    void plotGrowthCurvesComparison(const std::vector<double>& linear_pred,
                              const std::vector<double>& exp_pred,
                              const std::vector<double>& gomp_pred,
                              const std::vector<double>& log_pred,
                              const std::vector<double>& poly2_pred,
                              const std::vector<double>& poly3_pred,
                              bool linear_valid,
                              bool exp_valid,
                              bool gomp_valid,
                              bool logistic_valid,
                              bool poly2_valid,
                              bool poly3_valid) const;

public:
    SeriesControl(const RecordClass& recordData, const std::string& commodityName);
    SeriesClass createSeriesFromRecord(const RecordClass& recordData, const std::string& commodityName);
    const SeriesClass& getSeries() const;
    void processAnomalies(double criticalValue = 1.5);
    SeriesClass movingAverage5() const;
    SeriesClass weightedMovingAverage5() const;
    SeriesClass weightedMovingAverage7() const;
    SeriesClass chronologicalAverage12() const;
    SeriesClass exponentialSmoothing(double alpha = 0.3) const;
    void analyzeTrends() const;
    void displaySeriesInfo() const;
    void displayComparison(const std::vector<SeriesClass>& smoothedSeries) const;
    void plotAllSeries(const std::vector<SeriesClass>& smoothedSeries) const;
    void plotIndividualComparison(const SeriesClass& smoothedSeries) const;
    void plotDecomposition(int period = 12) const;
    void analyzeResiduals() const;
    void plotResiduals(const SeriesClass& residuals) const;
    void analyzeGrowthCurves() const;
    void forecastAnalysis(size_t shortStep = 3, size_t longStep = 7) const;
    void forecastAnalysisBrown(double alpha = 0.3, double train_fraction = 0.8);
    void plotForecastBrown(const SeriesClass::ForecastBacktestResult& result) const;
};

#endif