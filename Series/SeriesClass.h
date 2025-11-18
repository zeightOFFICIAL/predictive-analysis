#ifndef SERIESCLASS_H
#define SERIESCLASS_H

#include <vector>
#include <string>

class SeriesClass {
public:
    
    struct DecompositionResult {
        std::vector<double> original;
        std::vector<double> preliminaryTrend;
        std::vector<double> primaryTrend;
        std::vector<double> secondaryTrend;
        std::vector<double> finalTrend;
        std::vector<double> preliminarySeasonal;
        std::vector<double> primarySeasonal;
        std::vector<double> secondarySeasonal;
        std::vector<double> finalSeasonal;
        std::vector<double> preliminaryResidual;
        std::vector<double> primaryResidual;
        std::vector<double> secondaryResidual;
        std::vector<double> finalResidual;
    };
    struct ResidualAnalysisResult {
        
        size_t turningPointsCount;
        double turningPointsStatistic;
        bool isRandomByTurningPoints;
        
        size_t seriesCount;
        double seriesStatistic;
        bool isRandomBySeries;
        
        
        double skewness;
        double kurtosis;
        bool isNormalByMoments;
        
        double rSStatistic;
        bool isNormalByRS;
        
        
        double mean;
        double tStatistic;
        bool hasZeroMean;
        
        
        double durbinWatsonStatistic;
        bool isIndependent;
        
        
        bool isAdequate;
        std::string conclusion;
    };
struct Forecast {
    double point;
    double lower;
    double upper;
};
struct ForecastBacktestResult {
        std::vector<double> fixed_points;
        std::vector<double> adaptive_points;
        double mse_fixed = 0.0;
        double mse_adapt = 0.0;
        Forecast last_fixed_interval;
        Forecast last_adaptive_interval;
    };

    ForecastBacktestResult backtestExponential(double alpha, double train_fraction) const;
    std::vector<Forecast> forecastExponential(size_t steps, double alpha, double confidence_level = 0.95) const;
private:
    std::vector<double> data;
    std::vector<std::string> timestamps;
    std::string name;

    double calculateSkewness() const;
    double calculateKurtosis() const;
    double calculateRSStatistic() const;
    double calculateDurbinWatson() const;
    std::pair<size_t, bool> analyzeTurningPoints() const;
    std::pair<size_t, bool> analyzeSeriesTest() const;

    

public:
    
    SeriesClass(const std::vector<double>& values, 
                const std::vector<std::string>& times, 
                const std::string& seriesName);

    
    const std::vector<double>& getData() const;
    const std::vector<std::string>& getTimestamps() const;
    std::string getName() const;
    size_t size() const;
    void replaceData(const std::vector<double>& newData);

    
    std::vector<size_t> detectAnomaliesIrwin(double criticalValue = 1.5) const;
    void interpolateAnomalies(const std::vector<size_t>& anomalyIndices);

    
    std::vector<double> movingAverage(size_t window) const;
    std::vector<double> weightedMovingAverage(const std::vector<double>& weights) const;
    std::vector<double> exponentialSmoothing(double alpha) const;

    
    static std::vector<double> generateLinearWeights(size_t n);
    static std::vector<double> generateTriangularWeights(size_t n);

    
    std::pair<bool, double> checkTrendMeanDifferences() const;
    std::pair<bool, double> checkTrendFosterStewart() const;

    
    DecompositionResult decomposeTimeSeries(int period) const;

    
    std::vector<double> smoothThreePoints() const;
    std::vector<double> calculateFirstDifferences() const;
    std::vector<double> calculateSecondDifferences() const;
    std::vector<double> calculateRelativeFirstDifferences() const;
    std::vector<double> calculateLogFirstDifferences() const;
    std::vector<double> calculateGompertzIndicator() const;
    std::vector<double> calculateLogisticIndicator() const;
    
    
    std::pair<double, double> fitLinearPolynomial() const;
    std::pair<double, double> fitExponential() const;
    std::pair<double, double> fitGompertz() const;
    std::pair<double, double> fitLogistic() const;
    std::vector<double> fitPolynomial2() const;
    std::vector<double> fitPolynomial3() const;
    
    
    std::vector<double> predictLinear(double a, double b) const;
    std::vector<double> predictExponential(double a, double b) const;   
    
    void analyzeGrowthCurveCharacteristics() const;

    std::vector<double> predictGompertz(double k, double t0) const;
    std::vector<double> predictLogistic(double k, double A) const;
    std::vector<double> predictPolynomial2(const std::vector<double>& coeffs) const;
    std::vector<double> predictPolynomial3(const std::vector<double>& coeffs) const;

    ResidualAnalysisResult analyzeResiduals() const;



    std::vector<Forecast> forecastLinear(size_t steps, double alpha = 0.05) const;
    std::vector<Forecast> forecastPolynomial2(size_t steps, double alpha = 0.05) const;
    std::vector<Forecast> forecastPolynomial3(size_t steps, double alpha = 0.05) const;
    double getTCritical(double alpha, size_t df) const;


    ForecastBacktestResult backtestLinear(double train_fraction = 0.8) const;
    ForecastBacktestResult backtestPolynomial3(double train_fraction = 0.8) const;

};


#endif