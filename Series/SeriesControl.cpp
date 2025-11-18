#include "SeriesControl.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <cstdlib>
#include <filesystem>
#include <math.h>

SeriesControl::SeriesControl(const RecordClass& recordData, const std::string& commodityName)
    : series(createSeriesFromRecord(recordData, commodityName)) {
}

SeriesClass SeriesControl::createSeriesFromRecord(const RecordClass& recordData, const std::string& commodityName) {
    auto prices = recordData.getAllPrices(commodityName);
    std::vector<double> values;
    std::vector<std::string> timestamps;
    
    for (const auto& [date, price] : prices) {
        values.push_back(static_cast<double>(price));
        timestamps.push_back(recordData.formatDate(date));
    }
    
    std::string simplifiedName = simplifyCommodityName(commodityName);
    
    return SeriesClass(values, timestamps, simplifiedName);
}

std::string SeriesControl::simplifyCommodityName(const std::string& fullName) const {
    if (fullName == RecordClass::WTI_OIL) return "Crude Oil";
    else if (fullName == RecordClass::GOLD) return "Gold";
    else if (fullName == RecordClass::SILVER) return "Silver";
    else if (fullName == RecordClass::NATURAL_GAS) return "Natural Gas";
    else if (fullName == RecordClass::CORN) return "Corn";
    else if (fullName == RecordClass::WHEAT) return "Wheat";
    else if (fullName == RecordClass::SOYBEAN) return "Soybean";
    else if (fullName == RecordClass::COPPER) return "Copper";
    else if (fullName == RecordClass::PLATINUM) return "Platinum";
    else if (fullName == RecordClass::PALLADIUM) return "Palladium";
    else return fullName;
}

void SeriesControl::processAnomalies(double criticalValue) {
    std::cout << "=== ANOMALY PROCESSING ===" << std::endl;
    std::cout << "Original series: " << series.getName() << std::endl;
    std::cout << "Number of points: " << series.size() << std::endl;
    
    auto anomalies = series.detectAnomaliesIrwin(criticalValue);
    
    auto data = series.getData();
    if (data.size() >= 2) {
        double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
        double sumSq = 0.0;
        for (double value : data) {
            sumSq += (value - mean) * (value - mean);
        }
        double sigma = std::sqrt(sumSq / (data.size() - 1));
        
        std::cout << "--------------------------------" << std::endl;
        std::cout << "FINAL SUMMARY:" << std::endl;
        std::cout << "Standard Deviation (sigma): " << sigma << std::endl;
        std::cout << "Critical Value: " << criticalValue << std::endl;
        std::cout << "Anomalies detected: " << anomalies.size() << std::endl;
        std::cout << "=================================" << std::endl;
    }
    
    if (!anomalies.empty()) {
        std::cout << "Anomaly indices: ";
        for (size_t idx : anomalies) {
            std::cout << idx << " ";
        }
        std::cout << std::endl;
        
        auto originalData = series.getData();
        
        series.interpolateAnomalies(anomalies);
        
        std::cout << "Anomalies successfully interpolated" << std::endl;
    }
    std::cout << std::endl;
}

SeriesClass SeriesControl::movingAverage5() const {
    auto smoothedData = series.movingAverage(5);
    auto timestamps = series.getTimestamps();
    
    SeriesClass result(smoothedData, timestamps, series.getName() + " - MA5");
    return result;
}

SeriesClass SeriesControl::weightedMovingAverage5() const {
    auto weights = SeriesClass::generateLinearWeights(5);
    auto smoothedData = series.weightedMovingAverage(weights);
    auto timestamps = series.getTimestamps();
    
    SeriesClass result(smoothedData, timestamps, series.getName() + " - WMA5");
    return result;
}

SeriesClass SeriesControl::weightedMovingAverage7() const {
    auto weights = SeriesClass::generateLinearWeights(7);
    auto smoothedData = series.weightedMovingAverage(weights);
    auto timestamps = series.getTimestamps();
    
    SeriesClass result(smoothedData, timestamps, series.getName() + " - WMA7");
    return result;
}

SeriesClass SeriesControl::chronologicalAverage12() const {
    auto weights = SeriesClass::generateTriangularWeights(12);
    auto smoothedData = series.weightedMovingAverage(weights);
    auto timestamps = series.getTimestamps();
    
    SeriesClass result(smoothedData, timestamps, series.getName() + " - CA12");
    return result;
}

SeriesClass SeriesControl::exponentialSmoothing(double alpha) const {
    auto smoothedData = series.exponentialSmoothing(alpha);
    auto timestamps = series.getTimestamps();
    
    SeriesClass result(smoothedData, timestamps, series.getName() + " - Exp");
    return result;
}

void SeriesControl::displaySeriesInfo() const {
    std::cout << "=== SERIES INFORMATION ===" << std::endl;
    std::cout << "Name: " << series.getName() << std::endl;
    std::cout << "Number of points: " << series.size() << std::endl;
    
    if (series.size() > 0) {
        auto data = series.getData();
        double min = *std::min_element(data.begin(), data.end());
        double max = *std::max_element(data.begin(), data.end());
        double sum = std::accumulate(data.begin(), data.end(), 0.0);
        double mean = sum / data.size();
        
        std::cout << "Minimum value: " << min << std::endl;
        std::cout << "Maximum value: " << max << std::endl;
        std::cout << "Mean value: " << mean << std::endl;
        std::cout << "Date range: " << series.getTimestamps().front() 
                  << " - " << series.getTimestamps().back() << std::endl;
    }
    std::cout << std::endl;
}

void SeriesControl::displayComparison(const std::vector<SeriesClass>& smoothedSeries) const {
    std::cout << "=== SERIES COMPARISON ===" << std::endl;
    std::cout << std::left << std::setw(12) << "Date" 
              << std::setw(12) << "Original";
    
    for (const auto& s : smoothedSeries) {
        std::cout << std::setw(15) << s.getName();
    }
    std::cout << std::endl;
    
    std::cout << std::string(12 + 12 + smoothedSeries.size() * 15, '-') << std::endl;
    
    auto timestamps = series.getTimestamps();
    auto originalData = series.getData();
    
    size_t pointsToShow = std::min(size_t(20), timestamps.size());
    
    for (size_t i = 0; i < pointsToShow; ++i) {
        std::cout << std::setw(12) << timestamps[i]
                  << std::setw(12) << std::fixed << std::setprecision(4) << originalData[i];
        
        for (const auto& s : smoothedSeries) {
            std::cout << std::setw(15) << std::fixed << std::setprecision(4) << s.getData()[i];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

std::string SeriesControl::sanitizeFilename(const std::string& filename) const {
    std::string sanitized = filename;
    std::replace(sanitized.begin(), sanitized.end(), ' ', '_');
    std::replace(sanitized.begin(), sanitized.end(), '=', '_');
    std::replace(sanitized.begin(), sanitized.end(), '(', '_');
    std::replace(sanitized.begin(), sanitized.end(), ')', '_');
    std::replace(sanitized.begin(), sanitized.end(), '/', '_');
    std::replace(sanitized.begin(), sanitized.end(), '\\', '_');
    return sanitized;
}

void SeriesControl::saveDataToFile(const SeriesClass& series, const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create file " << filename << std::endl;
        return;
    }
    
    file << "# " << series.getName() << std::endl;
    file << "# Index Date Value" << std::endl;
    
    auto data = series.getData();
    auto timestamps = series.getTimestamps();
    
    for (size_t i = 0; i < data.size(); ++i) {
        file << i << " \"" << timestamps[i] << "\" " << std::fixed << std::setprecision(6) << data[i] << std::endl;
    }
    
    file.close();
}

void SeriesControl::cleanupDataFiles(const std::vector<std::string>& filenames) const {
    for (const auto& filename : filenames) {
        if (std::filesystem::exists(filename)) {
            std::filesystem::remove(filename);
        }
    }
}

void SeriesControl::plotAllSeries(const std::vector<SeriesClass>& smoothedSeries) const {
    std::cout << "=== GENERATING COMPARISON PLOT ===" << std::endl;
    
    std::filesystem::create_directories("plots/series");
    
    std::vector<std::string> dataFiles;
    
    std::string originalFile = "plots/series/original.dat";
    saveDataToFile(series, originalFile);
    dataFiles.push_back(originalFile);
    
    for (const auto& s : smoothedSeries) {
        std::string cleanName = sanitizeFilename(s.getName());
        std::string filename = "plots/series/" + cleanName + ".dat";
        saveDataToFile(s, filename);
        dataFiles.push_back(filename);
    }
    
    std::vector<std::string> residualFiles;
    auto originalData = series.getData();
    
    for (size_t i = 0; i < smoothedSeries.size(); ++i) {
        const auto& smoothed = smoothedSeries[i];
        auto smoothedData = smoothed.getData();
        std::string residualFilename = "plots/series/residual_" + sanitizeFilename(smoothed.getName()) + ".dat";
        
        std::ofstream residualFile(residualFilename);
        if (residualFile.is_open()) {
            residualFile << "# Residuals: " << smoothed.getName() << std::endl;
            residualFile << "# Index Date Residual" << std::endl;
            
            auto timestamps = series.getTimestamps();
            for (size_t j = 0; j < originalData.size() && j < smoothedData.size(); ++j) {
                double residual = originalData[j] - smoothedData[j];
                residualFile << j << " \"" << timestamps[j] << "\" " << std::fixed << std::setprecision(6) << residual << std::endl;
            }
            residualFile.close();
            residualFiles.push_back(residualFilename);
        }
    }
    
    std::string scriptFile = "plots/series/comparison_plot.gnu";
    std::ofstream script(scriptFile);
    script << "set terminal pngcairo size 1600,1200 enhanced font 'Arial,12'" << std::endl;
    script << "set output 'plots/series/all_smoothing_methods.png'" << std::endl;
    script << "set title 'Time Series: " << series.getName() << " - Smoothing Methods Comparison'" << std::endl;
    script << "set xlabel 'Time Index'" << std::endl;
    script << "set ylabel 'Price'" << std::endl;
    script << "set grid" << std::endl;
    script << "set key outside center top horizontal" << std::endl;
    script << "set multiplot layout 3,2 title 'Smoothing Methods Comparison'" << std::endl;
    script << std::endl;
    
    script << "set title 'Moving Average (5 points)'" << std::endl;
    script << "plot '" << originalFile << "' using 1:3 with lines lw 2 title 'Original', \\" << std::endl;
    script << "     'plots/series/" << sanitizeFilename(smoothedSeries[0].getName()) << ".dat' using 1:3 with lines lw 2 title '" 
           << smoothedSeries[0].getName() << "'" << std::endl;
    script << std::endl;
    
    script << "set title 'Weighted Moving Average (5 points)'" << std::endl;
    script << "plot '" << originalFile << "' using 1:3 with lines lw 2 title 'Original', \\" << std::endl;
    script << "     'plots/series/" << sanitizeFilename(smoothedSeries[1].getName()) << ".dat' using 1:3 with lines lw 2 title '" 
           << smoothedSeries[1].getName() << "'" << std::endl;
    script << std::endl;
    
    script << "set title 'Weighted Moving Average (7 points)'" << std::endl;
    script << "plot '" << originalFile << "' using 1:3 with lines lw 2 title 'Original', \\" << std::endl;
    script << "     'plots/series/" << sanitizeFilename(smoothedSeries[2].getName()) << ".dat' using 1:3 with lines lw 2 title '" 
           << smoothedSeries[2].getName() << "'" << std::endl;
    script << std::endl;
    
    script << "set title 'Chronological Average (12 points)'" << std::endl;
    script << "plot '" << originalFile << "' using 1:3 with lines lw 2 title 'Original', \\" << std::endl;
    script << "     'plots/series/" << sanitizeFilename(smoothedSeries[3].getName()) << ".dat' using 1:3 with lines lw 2 title '" 
           << smoothedSeries[3].getName() << "'" << std::endl;
    script << std::endl;
    
    script << "set title 'Exponential Smoothing'" << std::endl;
    script << "plot '" << originalFile << "' using 1:3 with lines lw 2 title 'Original', \\" << std::endl;
    script << "     'plots/series/" << sanitizeFilename(smoothedSeries[4].getName()) << ".dat' using 1:3 with lines lw 2 title '" 
           << smoothedSeries[4].getName() << "'" << std::endl;
    script << std::endl;
    
    script << "set title 'All Methods Comparison'" << std::endl;
    script << "plot '" << originalFile << "' using 1:3 with lines lw 3 title 'Original', \\" << std::endl;
    for (size_t i = 0; i < smoothedSeries.size(); ++i) {
        script << "     'plots/series/" << sanitizeFilename(smoothedSeries[i].getName()) << ".dat' using 1:3 with lines lw 2 title '" 
               << smoothedSeries[i].getName() << "'";
        if (i < smoothedSeries.size() - 1) script << ", \\";
        script << std::endl;
    }
    
    script << "unset multiplot" << std::endl;
    script.close();
    
    std::string command = "gnuplot \"" + scriptFile + "\"";
    int result = std::system(command.c_str());
    
    if (result == 0) {
        std::cout << "Multi-panel plot saved as: plots/series/all_smoothing_methods.png" << std::endl;
    } else {
        std::cerr << "Error: GNU Plot execution failed for multi-panel plot" << std::endl;
    }
    
    std::string overviewScriptFile = "plots/series/overview_plot.gnu";
    std::ofstream overviewScript(overviewScriptFile);
    overviewScript << "set terminal pngcairo size 6400,3600 enhanced font 'Arial,14'" << std::endl;
    overviewScript << "set output 'plots/series/smoothing_overview.png'" << std::endl;
    overviewScript << "set multiplot layout 2,1 title 'Time Series Overview: " << series.getName() << " - All Smoothing Methods and Residuals'" << std::endl;
    overviewScript << std::endl;
    
    overviewScript << "set title 'All Smoothing Methods Comparison' font 'Arial,16'" << std::endl;
    overviewScript << "set xlabel 'Time Index' font 'Arial,14'" << std::endl;
    overviewScript << "set ylabel 'Price' font 'Arial,14'" << std::endl;
    overviewScript << "set grid" << std::endl;
    overviewScript << "set key outside right top vertical box" << std::endl;
    overviewScript << "set key spacing 2" << std::endl;
    overviewScript << "set key width -15" << std::endl;
    overviewScript << "set key font ',12'" << std::endl;
    overviewScript << "set xrange [*:*]" << std::endl;
    overviewScript << "set yrange [*:*]" << std::endl;
    overviewScript << std::endl;
    
    overviewScript << "set style line 1 lw 1 lc rgb '#000000' dt 1" << std::endl;
    overviewScript << "set style line 2 lw 1 lc rgb '#FF0000' dt 1" << std::endl; 
    overviewScript << "set style line 3 lw 1 lc rgb '#00AA00' dt 1" << std::endl; 
    overviewScript << "set style line 4 lw 1 lc rgb '#0000FF' dt 1" << std::endl; 
    overviewScript << "set style line 5 lw 1 lc rgb '#FF00FF' dt 1" << std::endl; 
    overviewScript << "set style line 6 lw 1 lc rgb '#FF8C00' dt 1" << std::endl; 
    
    overviewScript << "plot '" << originalFile << "' using 1:3 with lines ls 1 title 'Original', \\" << std::endl;
    overviewScript << "     'plots/series/" << sanitizeFilename(smoothedSeries[0].getName()) << ".dat' using 1:3 with lines ls 2 title 'MA5', \\" << std::endl;
    overviewScript << "     'plots/series/" << sanitizeFilename(smoothedSeries[1].getName()) << ".dat' using 1:3 with lines ls 3 title 'WMA5', \\" << std::endl;
    overviewScript << "     'plots/series/" << sanitizeFilename(smoothedSeries[2].getName()) << ".dat' using 1:3 with lines ls 4 title 'WMA7', \\" << std::endl;
    overviewScript << "     'plots/series/" << sanitizeFilename(smoothedSeries[3].getName()) << ".dat' using 1:3 with lines ls 5 title 'CA12', \\" << std::endl;
    overviewScript << "     'plots/series/" << sanitizeFilename(smoothedSeries[4].getName()) << ".dat' using 1:3 with lines ls 6 title 'Exp'" << std::endl;
    overviewScript << std::endl;
    
    overviewScript << "set title 'Residuals (Original - Smoothed)' font 'Arial,16'" << std::endl;
    overviewScript << "set xlabel 'Time Index' font 'Arial,14'" << std::endl;
    overviewScript << "set ylabel 'Residual Value' font 'Arial,14'" << std::endl;
    overviewScript << "set grid" << std::endl;
    overviewScript << "set key outside right top vertical box" << std::endl;
    overviewScript << "set key spacing 2" << std::endl;
    overviewScript << "set key width -15" << std::endl;
    overviewScript << "set key font ',12'" << std::endl;
    overviewScript << "set xrange [GPVAL_X_MIN:GPVAL_X_MAX]" << std::endl;
    overviewScript << std::endl;
    
    overviewScript << "plot 'plots/series/residual_" << sanitizeFilename(smoothedSeries[0].getName()) << ".dat' using 1:3 with lines ls 2 lw 0.5 title 'MA5 Residuals', \\" << std::endl;
    overviewScript << "     'plots/series/residual_" << sanitizeFilename(smoothedSeries[1].getName()) << ".dat' using 1:3 with lines ls 3 lw 0.5 title 'WMA5 Residuals', \\" << std::endl;
    overviewScript << "     'plots/series/residual_" << sanitizeFilename(smoothedSeries[2].getName()) << ".dat' using 1:3 with lines ls 4 lw 0.5 title 'WMA7 Residuals', \\" << std::endl;
    overviewScript << "     'plots/series/residual_" << sanitizeFilename(smoothedSeries[3].getName()) << ".dat' using 1:3 with lines ls 5 lw 0.5 title 'CA12 Residuals', \\" << std::endl;
    overviewScript << "     'plots/series/residual_" << sanitizeFilename(smoothedSeries[4].getName()) << ".dat' using 1:3 with lines ls 6 lw 0.5 title 'Exp Residuals'" << std::endl;
    
    overviewScript << "unset multiplot" << std::endl;
    overviewScript.close();
    
    std::string overviewCommand = "gnuplot \"" + overviewScriptFile + "\"";
    int overviewResult = std::system(overviewCommand.c_str());
    
    if (overviewResult == 0) {
        std::cout << "Overview plot with residuals saved as: plots/series/smoothing_overview.png" << std::endl;
    } else {
        std::cerr << "Error: GNU Plot execution failed for overview plot" << std::endl;
    }
    
    std::string individualResidualsScriptFile = "plots/series/individual_residuals_plot.gnu";
    std::ofstream individualResidualsScript(individualResidualsScriptFile);
    individualResidualsScript << "set terminal pngcairo size 6400,3600 enhanced font 'Arial,14'" << std::endl;
    individualResidualsScript << "set output 'plots/series/individual_residuals.png'" << std::endl;
    individualResidualsScript << "set multiplot layout 3,2 title 'Individual Residuals Analysis: " << series.getName() << "'" << std::endl;
    individualResidualsScript << std::endl;
    
    for (size_t i = 0; i < smoothedSeries.size(); ++i) {
        individualResidualsScript << "set title '" << smoothedSeries[i].getName() << " Residuals' font 'Arial,16'" << std::endl;
        individualResidualsScript << "set xlabel 'Time Index' font 'Arial,14'" << std::endl;
        individualResidualsScript << "set ylabel 'Residual Value' font 'Arial,14'" << std::endl;
        individualResidualsScript << "set grid" << std::endl;
        individualResidualsScript << "plot 'plots/series/residual_" << sanitizeFilename(smoothedSeries[i].getName()) << ".dat' using 1:3 with lines lw 1 lc rgb '" 
                                  << getColorByIndex(i) << "' title 'Residuals'" << std::endl;
        individualResidualsScript << std::endl;
    }
    
    individualResidualsScript << "unset multiplot" << std::endl;
    individualResidualsScript.close();
    
    std::string individualResidualsCommand = "gnuplot \"" + individualResidualsScriptFile + "\"";
    int individualResult = std::system(individualResidualsCommand.c_str());
    
    if (individualResult == 0) {
        std::cout << "Individual residuals plot saved as: plots/series/individual_residuals.png" << std::endl;
    } else {
        std::cerr << "Error: GNU Plot execution failed for individual residuals plot" << std::endl;
    }
    
    cleanupDataFiles(dataFiles);
    cleanupDataFiles(residualFiles);
    if (std::filesystem::exists(scriptFile)) {
        std::filesystem::remove(scriptFile);
    }
    if (std::filesystem::exists(overviewScriptFile)) {
        std::filesystem::remove(overviewScriptFile);
    }
    if (std::filesystem::exists(individualResidualsScriptFile)) {
        std::filesystem::remove(individualResidualsScriptFile);
    }
}

std::string SeriesControl::getColorByIndex(size_t index) const {
    switch (index) {
        case 0: return "#FF0000";
        case 1: return "#00AA00";
        case 2: return "#0000FF"; 
        case 3: return "#FF00FF";
        case 4: return "#FF8C00"; 
        default: return "#000000";
    }
}

void SeriesControl::plotIndividualComparison(const SeriesClass& smoothedSeries) const {
    std::string seriesName = smoothedSeries.getName();
    std::string cleanName = sanitizeFilename(seriesName);
    std::string filenameBase = "plots/series/individual_" + cleanName;
    
    std::string originalFile = filenameBase + "_original.dat";
    std::string smoothedFile = filenameBase + "_smoothed.dat";
    
    saveDataToFile(series, originalFile);
    saveDataToFile(smoothedSeries, smoothedFile);
    
    std::string scriptFile = filenameBase + ".gnu";
    std::ofstream script(scriptFile);
    script << "set terminal pngcairo size 1200,800 enhanced font 'Arial,12'" << std::endl;
    script << "set output '" << filenameBase << ".png'" << std::endl;
    script << "set title 'Time Series: " << series.getName() << " vs " << seriesName << "'" << std::endl;
    script << "set xlabel 'Time Index'" << std::endl;
    script << "set ylabel 'Price'" << std::endl;
    script << "set grid" << std::endl;
    script << "set key outside top center horizontal" << std::endl;
    script << "plot '" << originalFile << "' using 1:3 with lines lw 2 title 'Original Data', \\" << std::endl;
    script << "     '" << smoothedFile << "' using 1:3 with lines lw 2 title '" << seriesName << "'" << std::endl;
    script.close();
    
    std::string command = "gnuplot \"" + scriptFile + "\"";
    int result = std::system(command.c_str());
    
    if (result == 0) {
        std::cout << "Individual plot saved as: " << filenameBase << ".png" << std::endl;
    } else {
        std::cerr << "Error: GNU Plot execution failed for individual plot" << std::endl;
    }
    
    cleanupDataFiles({originalFile, smoothedFile});
    if (std::filesystem::exists(scriptFile)) {
        std::filesystem::remove(scriptFile);
    }
}

const SeriesClass& SeriesControl::getSeries() const {
    return series;
}

void SeriesControl::analyzeTrends() const {
    std::cout << "=== TREND ANALYSIS ===" << std::endl;
    std::cout << "Series: " << series.getName() << std::endl;
    std::cout << "Number of points: " << series.size() << std::endl;
    
    std::cout << "\n1. MEAN DIFFERENCES METHOD:" << std::endl;
    auto [has_trend_mean, t_statistic] = series.checkTrendMeanDifferences();
    
    std::cout << "Trend detected: " << (has_trend_mean ? "YES" : "NO") << std::endl;
    
    if (has_trend_mean) {
        std::cout << "Trend direction: " << (t_statistic > 0 ? "DECREASING" : "INCREASING") << std::endl;
    }
    
    std::cout << "\n2. FOSTER-STEWART METHOD:" << std::endl;
    auto [has_trend_foster, foster_statistic] = series.checkTrendFosterStewart();
    
    std::cout << "Trend detected: " << (has_trend_foster ? "YES" : "NO") << std::endl;
    
    std::cout << "\n=== FINAL CONCLUSION ===" << std::endl;
    if (has_trend_mean && has_trend_foster) {
        std::cout << "STRONG EVIDENCE of trend presence" << std::endl;
    } else if (has_trend_mean || has_trend_foster) {
        std::cout << "WEAK EVIDENCE of trend presence" << std::endl;
    } else {
        std::cout << "NO EVIDENCE of trend presence" << std::endl;
    }
    std::cout << "=================================" << std::endl << std::endl;
}

void SeriesControl::plotDecomposition(int period) const {
    std::cout << "=== TIME SERIES DECOMPOSITION ===" << std::endl;
    std::cout << "Series: " << series.getName() << std::endl;
    std::cout << "Number of points: " << series.size() << std::endl;
    std::cout << "Period: " << period << std::endl;
    
    auto decomposition = series.decomposeTimeSeries(period);
    
    std::filesystem::create_directories("plots/decomposition");
    
    std::vector<std::string> dataFiles;
    
    std::string trendFile = "plots/decomposition/trend.dat";
    std::ofstream trend(trendFile);
    trend << "# Index Original PreliminaryTrend PrimaryTrend FinalTrend" << std::endl;
    for (size_t i = 0; i < decomposition.original.size(); ++i) {
        trend << i << " " << std::fixed << std::setprecision(6) 
              << decomposition.original[i] << " "
              << decomposition.preliminaryTrend[i] << " "
              << decomposition.primaryTrend[i] << " "
              << decomposition.finalTrend[i] << std::endl;
    }
    trend.close();
    dataFiles.push_back(trendFile);
    
    std::string seasonalFile = "plots/decomposition/seasonal.dat";
    std::ofstream seasonal(seasonalFile);
    seasonal << "# Index PrimarySeasonal FinalSeasonal" << std::endl;
    for (size_t i = 0; i < decomposition.original.size(); ++i) {
        seasonal << i << " " << std::fixed << std::setprecision(6) 
                 << decomposition.primarySeasonal[i] << " "
                 << decomposition.finalSeasonal[i] << std::endl;
    }
    seasonal.close();
    dataFiles.push_back(seasonalFile);
    
    std::string residualFile = "plots/decomposition/residual.dat";
    std::ofstream residual(residualFile);
    residual << "# Index FinalResidual" << std::endl;
    for (size_t i = 0; i < decomposition.original.size(); ++i) {
        residual << i << " " << std::fixed << std::setprecision(6) 
                 << decomposition.finalResidual[i] << std::endl;
    }
    residual.close();
    dataFiles.push_back(residualFile);
    
    std::string scriptFile = "plots/decomposition/decomposition_plot.gnu";
    std::ofstream script(scriptFile);
    
    script << "set terminal pngcairo size 7200,5400 enhanced font 'Arial,20'" << std::endl;
    script << "set output 'plots/decomposition/time_series_decomposition.png'" << std::endl;
    script << "set encoding utf8" << std::endl;
    script << "set multiplot layout 3,1 title 'Time Series Decomposition: " << series.getName() << " (Y_t = U_t + V_t + ε_t)'" << std::endl;
    script << "set lmargin 15" << std::endl;
    script << "set rmargin 15" << std::endl;
    script << "set tmargin 5" << std::endl;
    script << "set bmargin 5" << std::endl;
    script << std::endl;
    
    script << "set title '1. Original Series and Trend Components' font 'Arial,28'" << std::endl;
    script << "set xlabel 'Time Index' font 'Arial,24'" << std::endl;
    script << "set ylabel 'Value' font 'Arial,24'" << std::endl;
    script << "set grid linecolor '#DDDDDD' linewidth 0.5" << std::endl;
    script << "set key outside top center horizontal font 'Arial,20'" << std::endl;
    script << "set key spacing 2" << std::endl;
    script << "set yrange [*:*]" << std::endl;
    script << "set linetype 1 lw 1.0 lc rgb '#000000'  # Original - black solid" << std::endl;
    script << "set linetype 2 lw 1.0 lc rgb '#FF0000'  # Preliminary - red solid" << std::endl;
    script << "set linetype 3 lw 1.0 lc rgb '#00AA00'  # Primary - green solid" << std::endl;
    script << "set linetype 4 lw 2.0 lc rgb '#FF8C00'  # Final - orange solid" << std::endl;
    script << "plot '" << trendFile << "' using 1:2 with lines linetype 1 title 'Original (Y_t)', \\" << std::endl;
    script << "     '" << trendFile << "' using 1:3 with lines linetype 2 title 'Preliminary Trend (MA30)', \\" << std::endl;
    script << "     '" << trendFile << "' using 1:4 with lines linetype 3 title 'Primary Trend (WMA7)', \\" << std::endl;
    script << "     '" << trendFile << "' using 1:5 with lines linetype 4 title 'Final Trend (Centered MA)'" << std::endl;
    script << std::endl;
    
    script << "set title '2. Seasonal Components (V_t)' font 'Arial,28'" << std::endl;
    script << "set xlabel 'Time Index' font 'Arial,24'" << std::endl;
    script << "set ylabel 'Seasonal Value' font 'Arial,24'" << std::endl;
    script << "set grid linecolor '#DDDDDD' linewidth 0.5" << std::endl;
    script << "set key outside top center horizontal font 'Arial,20'" << std::endl;
    script << "set key spacing 2" << std::endl;
    script << "set yrange [*:*]" << std::endl;
    script << "set linetype 5 lw 1.0 lc rgb '#00AA00'  # Primary Seasonal - green solid" << std::endl;
    script << "set linetype 6 lw 1.5 lc rgb '#0000FF'  # Final Seasonal - blue solid" << std::endl;
    script << "plot '" << seasonalFile << "' using 1:2 with lines linetype 5 title 'Primary Seasonal', \\" << std::endl;
    script << "     '" << seasonalFile << "' using 1:3 with lines linetype 6 title 'Final Seasonal (V_t)'" << std::endl;
    script << std::endl;
    
    script << "set title '3. Residual Component (ε_t)' font 'Arial,28'" << std::endl;
    script << "set xlabel 'Time Index' font 'Arial,24'" << std::endl;
    script << "set ylabel 'Residual Value' font 'Arial,24'" << std::endl;
    script << "set grid linecolor '#DDDDDD' linewidth 0.5" << std::endl;
    script << "set key outside top center horizontal font 'Arial,20'" << std::endl;
    script << "set key spacing 2" << std::endl;
    script << "set yrange [*:*]" << std::endl;
    script << "set linetype 7 lw 1.0 lc rgb '#8B4513'  # Residual - brown solid" << std::endl;
    script << "plot '" << residualFile << "' using 1:2 with lines linetype 7 title 'Final Residual (ε_t)'" << std::endl;
    script << std::endl;
    
    script << "unset multiplot" << std::endl;
    script.close();
    
    std::string command = "gnuplot \"" + scriptFile + "\"";
    int result = std::system(command.c_str());
    
    if (result == 0) {
        std::cout << "Decomposition plot saved as: plots/decomposition/time_series_decomposition.png" << std::endl;
        std::cout << "Resolution: 7200x5400 pixels" << std::endl;
        std::cout << "All lines are SOLID" << std::endl;
    } else {
        std::cerr << "Error: GNU Plot execution failed for decomposition plot" << std::endl;
    }
    
    cleanupDataFiles(dataFiles);
    if (std::filesystem::exists(scriptFile)) {
        std::filesystem::remove(scriptFile);
    }
    
    std::cout << "\n=== SEASONAL COMPONENT ANALYSIS ===" << std::endl;
    
    std::vector<double> uniquePrimarySeasonal;
    std::vector<double> uniqueFinalSeasonal;
    
    for (int i = 0; i < period; ++i) {
        uniquePrimarySeasonal.push_back(decomposition.primarySeasonal[i]);
        uniqueFinalSeasonal.push_back(decomposition.finalSeasonal[i]);
    }
    
    auto [minPrimary, maxPrimary] = std::minmax_element(uniquePrimarySeasonal.begin(), uniquePrimarySeasonal.end());
    auto [minFinal, maxFinal] = std::minmax_element(uniqueFinalSeasonal.begin(), uniqueFinalSeasonal.end());
    
    double primaryMean = std::accumulate(uniquePrimarySeasonal.begin(), uniquePrimarySeasonal.end(), 0.0) / period;
    double finalMean = std::accumulate(uniqueFinalSeasonal.begin(), uniqueFinalSeasonal.end(), 0.0) / period;
    
    std::cout << "Primary Seasonal pattern:" << std::endl;
    std::cout << "  Range: [" << *minPrimary << ", " << *maxPrimary << "]" << std::endl;
    std::cout << "  Amplitude: " << (*maxPrimary - *minPrimary) << std::endl;
    std::cout << "  Mean: " << primaryMean << std::endl;
    
    std::cout << "Final Seasonal pattern:" << std::endl;
    std::cout << "  Range: [" << *minFinal << ", " << *maxFinal << "]" << std::endl;
    std::cout << "  Amplitude: " << (*maxFinal - *minFinal) << std::endl;
    std::cout << "  Mean: " << finalMean << std::endl;
    
    double seasonalDiff = 0.0;
    for (int i = 0; i < period; ++i) {
        seasonalDiff += std::abs(uniquePrimarySeasonal[i] - uniqueFinalSeasonal[i]);
    }
    seasonalDiff /= period;
    
    std::cout << "Average difference between seasonal components: " << seasonalDiff << std::endl;
    
    std::cout << "First 5 primary seasonal values: ";
    for (int i = 0; i < std::min(5, period); ++i) {
        std::cout << uniquePrimarySeasonal[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "First 5 final seasonal values: ";
    for (int i = 0; i < std::min(5, period); ++i) {
        std::cout << uniqueFinalSeasonal[i] << " ";
    }
    std::cout << std::endl;
}


void SeriesControl::analyzeGrowthCurves() const {
    std::cout << "=== GROWTH CURVE ANALYSIS ===" << std::endl;
    std::cout << "Series: " << series.getName() << std::endl;
    
    
    auto [a_linear, b_linear] = series.fitLinearPolynomial();
    auto [a_exp, b_exp] = series.fitExponential();
    auto [a_gomp, b_gomp] = series.fitGompertz();
    auto [a_log, A_log] = series.fitLogistic();
    auto poly2_coeffs = series.fitPolynomial2();
    auto poly3_coeffs = series.fitPolynomial3();
    
    std::cout << "\n=== FITTED PARAMETERS ===" << std::endl;
    std::cout << "Linear: y = " << a_linear << " + " << b_linear << " * t" << std::endl;
    std::cout << "Exponential: y = " << a_exp << " * exp(" << b_exp << " * t)" << std::endl;
    std::cout << "Gompertz: k = " << a_gomp << ", t0 = " << b_gomp << std::endl;
    std::cout << "Logistic: k = " << a_log << ", A = " << A_log << std::endl;
    
    if (poly2_coeffs.size() == 3) {
        std::cout << "Poly2: y = " << poly2_coeffs[0] << " + " << poly2_coeffs[1] 
                  << " * t + " << poly2_coeffs[2] << " * t2" << std::endl;
    }
    
    if (poly3_coeffs.size() == 4) {
        std::cout << "Poly3: y = " << poly3_coeffs[0] << " + " << poly3_coeffs[1] 
                  << " * t + " << poly3_coeffs[2] << " * t2 + " << poly3_coeffs[3] << " * t3" << std::endl;
    }
    
    
    bool linear_valid = (std::abs(b_linear) > 1e-8 && std::abs(b_linear) < 1000.0);
    bool exp_valid = (std::abs(b_exp) > 1e-10 && std::abs(b_exp) < 1.0 && a_exp > 1.0);
    bool gomp_valid = (a_gomp > 1e-8 && a_gomp < 10.0 && std::abs(b_gomp) < series.size() * 100.0);
    bool logistic_valid = (a_log > 1e-8 && a_log < 10.0 && A_log > 0.01 && A_log < 10000.0);
    bool poly2_valid = (!poly2_coeffs.empty() && 
                       std::abs(poly2_coeffs[1]) < 1000.0 && 
                       std::abs(poly2_coeffs[2]) < 1000.0);
    bool poly3_valid = (!poly3_coeffs.empty() && 
                       std::abs(poly3_coeffs[1]) < 1000.0 && 
                       std::abs(poly3_coeffs[2]) < 1000.0 &&
                       std::abs(poly3_coeffs[3]) < 1000.0);
    
    
    auto data = series.getData();
    double data_mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    
    auto calculate_metrics = [&](const std::vector<double>& predictions) -> std::pair<double, double> {
        if (predictions.empty() || predictions.size() != data.size()) {
            return {0.0, 0.0};
        }
        
        double mse = 0.0;
        double ss_total = 0.0;
        double ss_residual = 0.0;
        
        for (size_t i = 0; i < data.size(); ++i) {
            double error = data[i] - predictions[i];
            mse += error * error;
            ss_total += (data[i] - data_mean) * (data[i] - data_mean);
            ss_residual += error * error;
        }
        
        mse /= data.size();
        double r_squared = (ss_total > 0) ? 1.0 - (ss_residual / ss_total) : 0.0;
        
        return {mse, r_squared};
    };
    
    
    auto linear_pred = linear_valid ? series.predictLinear(a_linear, b_linear) : std::vector<double>();
    auto exp_pred = exp_valid ? series.predictExponential(a_exp, b_exp) : std::vector<double>();
    auto gomp_pred = gomp_valid ? series.predictGompertz(a_gomp, b_gomp) : std::vector<double>();
    auto log_pred = logistic_valid ? series.predictLogistic(a_log, A_log) : std::vector<double>();
    auto poly2_pred = poly2_valid ? series.predictPolynomial2(poly2_coeffs) : std::vector<double>();
    auto poly3_pred = poly3_valid ? series.predictPolynomial3(poly3_coeffs) : std::vector<double>();
    
    auto [mse_linear, r2_linear] = calculate_metrics(linear_pred);
    auto [mse_exp, r2_exp] = calculate_metrics(exp_pred);
    auto [mse_gomp, r2_gomp] = calculate_metrics(gomp_pred);
    auto [mse_log, r2_log] = calculate_metrics(log_pred);
    auto [mse_poly2, r2_poly2] = calculate_metrics(poly2_pred);
    auto [mse_poly3, r2_poly3] = calculate_metrics(poly3_pred);
    
    
    std::cout << "\n=== MODEL QUALITY ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    
    if (linear_valid) {
        std::cout << "LINEAR - MSE: " << mse_linear << ", R-sqr: " << r2_linear << std::endl;
    }
    if (exp_valid) {
        std::cout << "EXPONENTIAL - MSE: " << mse_exp << ", R-sqr: " << r2_exp << std::endl;
    }
    if (gomp_valid) {
        std::cout << "GOMPERTZ - MSE: " << mse_gomp << ", R-sqr: " << r2_gomp << std::endl;
    }
    if (logistic_valid) {
        std::cout << "LOGISTIC - MSE: " << mse_log << ", R-sqr: " << r2_log << std::endl;
    }
    if (poly2_valid) {
        std::cout << "POLY2 - MSE: " << mse_poly2 << ", R-sqr: " << r2_poly2 << std::endl;
    }
    if (poly3_valid) {
        std::cout << "POLY3 - MSE: " << mse_poly3 << ", R-sqr: " << r2_poly3 << std::endl;
    }
    
    
    std::cout << "\n=== MODEL RANKING ===" << std::endl;
    std::vector<std::pair<double, std::string>> models_r2;
    if (linear_valid) models_r2.push_back({r2_linear, "LINEAR"});
    if (exp_valid) models_r2.push_back({r2_exp, "EXPONENTIAL"});
    if (gomp_valid) models_r2.push_back({r2_gomp, "GOMPERTZ"});
    if (logistic_valid) models_r2.push_back({r2_log, "LOGISTIC"});
    if (poly2_valid) models_r2.push_back({r2_poly2, "POLY2"});
    if (poly3_valid) models_r2.push_back({r2_poly3, "POLY3"});
    
    std::sort(models_r2.rbegin(), models_r2.rend());
    
    for (size_t i = 0; i < models_r2.size(); ++i) {
        std::cout << i + 1 << ". " << models_r2[i].second << " (R-sqr: " << models_r2[i].first << ")" << std::endl;
    }
    
    
    std::cout << "\n=== FINAL RECOMMENDATION ===" << std::endl;
    if (!models_r2.empty()) {
        std::string best_model = models_r2[0].second;
        double best_r2 = models_r2[0].first;
        
        std::cout << "BEST MODEL: " << best_model << std::endl;
        std::cout << "R-sqr: " << best_r2 << " (";
        
        if (best_r2 > 0.9) {
            std::cout << "EXCELLENT)";
        } else if (best_r2 > 0.8) {
            std::cout << "VERY GOOD)";
        } else if (best_r2 > 0.7) {
            std::cout << "GOOD)";
        } else if (best_r2 > 0.6) {
            std::cout << "MODERATE)";
        } else {
            std::cout << "WEAK)";
        }
        std::cout << std::endl;
        
        if (best_model == "POLY2" || best_model == "POLY3") {
            std::cout << "Note: Higher order polynomials may overfit" << std::endl;
        }
    }
    
    
    plotGrowthCurvesComparison(linear_pred, exp_pred, gomp_pred, log_pred, poly2_pred, poly3_pred,
                              linear_valid, exp_valid, gomp_valid, logistic_valid, poly2_valid, poly3_valid);
}

void SeriesControl::plotGrowthCurvesComparison(const std::vector<double>& linear_pred,
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
                                             bool poly3_valid) const {
    std::cout << "\nGenerating comparison plot..." << std::endl;
    
    std::filesystem::create_directories("plots/growth_curves");
    
    
    std::string originalFile = "plots/growth_curves/original.dat";
    saveGrowthCurveData(series, originalFile, "Original");
    
    std::vector<std::string> dataFiles = {originalFile};
    
    if (linear_valid && !linear_pred.empty()) {
        std::string linearFile = "plots/growth_curves/linear.dat";
        saveGrowthCurveData(linear_pred, series.getTimestamps(), linearFile, "Linear");
        dataFiles.push_back(linearFile);
    }
    
    if (exp_valid && !exp_pred.empty()) {
        std::string expFile = "plots/growth_curves/exponential.dat";
        saveGrowthCurveData(exp_pred, series.getTimestamps(), expFile, "Exponential");
        dataFiles.push_back(expFile);
    }
    
    if (gomp_valid && !gomp_pred.empty()) {
        std::string gompFile = "plots/growth_curves/gompertz.dat";
        saveGrowthCurveData(gomp_pred, series.getTimestamps(), gompFile, "Gompertz");
        dataFiles.push_back(gompFile);
    }
    
    if (logistic_valid && !log_pred.empty()) {
        std::string logFile = "plots/growth_curves/logistic.dat";
        saveGrowthCurveData(log_pred, series.getTimestamps(), logFile, "Logistic");
        dataFiles.push_back(logFile);
    }
    
    if (poly2_valid && !poly2_pred.empty()) {
        std::string poly2File = "plots/growth_curves/polynomial2.dat";
        saveGrowthCurveData(poly2_pred, series.getTimestamps(), poly2File, "Polynomial2");
        dataFiles.push_back(poly2File);
    }
    
    if (poly3_valid && !poly3_pred.empty()) {
        std::string poly3File = "plots/growth_curves/polynomial3.dat";
        saveGrowthCurveData(poly3_pred, series.getTimestamps(), poly3File, "Polynomial3");
        dataFiles.push_back(poly3File);
    }
    
    
    std::string scriptFile = "plots/growth_curves/growth_comparison.gnu";
    std::ofstream script(scriptFile);
    
    script << "set terminal pngcairo size 3200,2000 enhanced font 'Arial,14'" << std::endl;
    script << "set output 'plots/growth_curves/growth_curves_comparison.png'" << std::endl;
    script << "set title 'Growth Curves Comparison: " << series.getName() << "' font 'Arial,20'" << std::endl;
    script << "set xlabel 'Time Index' font 'Arial,16'" << std::endl;
    script << "set ylabel 'Value' font 'Arial,16'" << std::endl;
    script << "set grid" << std::endl;
    script << "set key outside center top horizontal font 'Arial,14'" << std::endl;
    script << "set key spacing 2" << std::endl;
    script << std::endl;
    
    
    script << "set style line 1 lw 1.0 lc rgb '#000000' dt 1" << std::endl;  
    script << "set style line 2 lw 1.0 lc rgb '#FF0000' dt 1" << std::endl;  
    script << "set style line 3 lw 1.0 lc rgb '#00AA00' dt 1" << std::endl;  
    script << "set style line 4 lw 1.0 lc rgb '#0000FF' dt 1" << std::endl;  
    script << "set style line 5 lw 1.0 lc rgb '#FF00FF' dt 1" << std::endl;  
    script << "set style line 6 lw 1.0 lc rgb '#FF8C00' dt 1" << std::endl;  
    script << "set style line 7 lw 1.0 lc rgb '#8A2BE2' dt 1" << std::endl;  
    script << std::endl;
    
    script << "plot '" << originalFile << "' using 1:3 with lines ls 1 title 'Original Data'";
    
    int line_style = 2;
    
    if (linear_valid && !linear_pred.empty()) {
        script << ", \\" << std::endl;
        script << "     'plots/growth_curves/linear.dat' using 1:3 with lines ls " << line_style << " title 'Linear'";
        line_style++;
    }
    
    if (exp_valid && !exp_pred.empty()) {
        script << ", \\" << std::endl;
        script << "     'plots/growth_curves/exponential.dat' using 1:3 with lines ls " << line_style << " title 'Exponential'";
        line_style++;
    }
    
    if (gomp_valid && !gomp_pred.empty()) {
        script << ", \\" << std::endl;
        script << "     'plots/growth_curves/gompertz.dat' using 1:3 with lines ls " << line_style << " title 'Gompertz'";
        line_style++;
    }
    
    if (logistic_valid && !log_pred.empty()) {
        script << ", \\" << std::endl;
        script << "     'plots/growth_curves/logistic.dat' using 1:3 with lines ls " << line_style << " title 'Logistic'";
        line_style++;
    }
    
    if (poly2_valid && !poly2_pred.empty()) {
        script << ", \\" << std::endl;
        script << "     'plots/growth_curves/polynomial2.dat' using 1:3 with lines ls " << line_style << " title 'Poly2'";
        line_style++;
    }
    
    if (poly3_valid && !poly3_pred.empty()) {
        script << ", \\" << std::endl;
        script << "     'plots/growth_curves/polynomial3.dat' using 1:3 with lines ls " << line_style << " title 'Poly3'";
    }
    
    script << std::endl;
    
    script.close();
    
    
    std::string command = "gnuplot \"" + scriptFile + "\"";
    int result = std::system(command.c_str());
    
    if (result == 0) {
        std::cout << "Plot saved: plots/growth_curves/growth_curves_comparison.png" << std::endl;
        std::cout << "Resolution: 3200x2000" << std::endl;
    } else {
        std::cerr << "Error: GNU Plot execution failed" << std::endl;
    }
    
    
    cleanupDataFiles(dataFiles);
    if (std::filesystem::exists(scriptFile)) {
        std::filesystem::remove(scriptFile);
    }
}

void SeriesControl::saveGrowthCurveData(const std::vector<double>& values, 
                                      const std::vector<std::string>& timestamps,
                                      const std::string& filename, 
                                      const std::string& name) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create file " << filename << std::endl;
        return;
    }
    
    file << "# " << name << std::endl;
    file << "# Index Date Value" << std::endl;
    
    for (size_t i = 0; i < values.size() && i < timestamps.size(); ++i) {
        file << i << " \"" << timestamps[i] << "\" " << std::fixed << std::setprecision(6) << values[i] << std::endl;
    }
    
    file.close();
}

void SeriesControl::saveGrowthCurveData(const SeriesClass& series, 
                                      const std::string& filename, 
                                      const std::string& name) const {
    saveGrowthCurveData(series.getData(), series.getTimestamps(), filename, name);
}

void SeriesControl::analyzeResiduals() const {
    std::cout << "=== RESIDUAL COMPONENT ANALYSIS ===" << std::endl;
    
    
    auto decomposition = series.decomposeTimeSeries(12); 
    SeriesClass residualsSeries(decomposition.finalResidual, 
                               series.getTimestamps(), 
                               series.getName() + " - Residuals");
    
    
    auto analysis = residualsSeries.analyzeResiduals();
    
    
    std::cout << "Residual series: " << residualsSeries.getName() << std::endl;
    std::cout << "Number of points: " << residualsSeries.size() << std::endl;
    
    std::cout << "\n1. RANDOMNESS TESTS:" << std::endl;
    std::cout << "Turning points count: " << analysis.turningPointsCount << std::endl;
    std::cout << "Random by turning points: " << (analysis.isRandomByTurningPoints ? "YES" : "NO") << std::endl;
    std::cout << "Series count: " << analysis.seriesCount << std::endl;
    std::cout << "Random by series test: " << (analysis.isRandomBySeries ? "YES" : "NO") << std::endl;
    
    std::cout << "\n2. NORMALITY TESTS:" << std::endl;
    std::cout << "Skewness: " << analysis.skewness << " (should be near 0)" << std::endl;
    std::cout << "Kurtosis: " << analysis.kurtosis << " (should be near 0)" << std::endl;
    std::cout << "Normal by moments: " << (analysis.isNormalByMoments ? "YES" : "NO") << std::endl;
    std::cout << "RS-statistic: " << analysis.rSStatistic << " (should be 2.5-3.5)" << std::endl;
    std::cout << "Normal by RS-test: " << (analysis.isNormalByRS ? "YES" : "NO") << std::endl;
    
    std::cout << "\n3. ZERO MEAN TEST:" << std::endl;
    std::cout << "Mean: " << analysis.mean << std::endl;
    std::cout << "T-statistic: " << analysis.tStatistic << " (should be < 2.0)" << std::endl;
    std::cout << "Has zero mean: " << (analysis.hasZeroMean ? "YES" : "NO") << std::endl;
    
    std::cout << "\n4. INDEPENDENCE TEST:" << std::endl;
    std::cout << "Durbin-Watson statistic: " << analysis.durbinWatsonStatistic 
              << " (should be 1.5-2.5)" << std::endl;
    std::cout << "Independent: " << (analysis.isIndependent ? "YES" : "NO") << std::endl;
    
    std::cout << "\n" << analysis.conclusion << std::endl;
    
    
    plotResiduals(residualsSeries);
}


void SeriesControl::plotResiduals(const SeriesClass& residuals) const {
    std::filesystem::create_directories("plots/residuals");
    
    std::string dataFile = "plots/residuals/residuals.dat";
    std::ofstream file(dataFile);
    
    file << "# Index Residual" << std::endl;
    auto data = residuals.getData();
    for (size_t i = 0; i < data.size(); ++i) {
        file << i << " " << std::fixed << std::setprecision(6) << data[i] << std::endl;
    }
    file.close();
    
    std::string scriptFile = "plots/residuals/residuals_plot.gnu";
    std::ofstream script(scriptFile);
    
    script << "set terminal pngcairo size 1600,800 enhanced font 'Arial,12'" << std::endl;
    script << "set output 'plots/residuals/residuals_analysis.png'" << std::endl;
    script << "set multiplot layout 2,2 title 'Residual Analysis: " << series.getName() << "'" << std::endl;
    
    
    script << "set title 'Residuals vs Time'" << std::endl;
    script << "set xlabel 'Time Index'" << std::endl;
    script << "set ylabel 'Residual Value'" << std::endl;
    script << "set grid" << std::endl;
    script << "plot '" << dataFile << "' using 1:2 with lines lw 1 lc rgb 'blue' title 'Residuals', \\" << std::endl;
    script << "     0 with lines lw 2 lc rgb 'red' title 'Zero line'" << std::endl;
    
    
    script << "set title 'Residuals Distribution'" << std::endl;
    script << "set xlabel 'Residual Value'" << std::endl;
    script << "set ylabel 'Frequency'" << std::endl;
    script << "set grid" << std::endl;
    script << "binwidth = 0.1" << std::endl;
    script << "bin(x,width) = width*floor(x/width)" << std::endl;
    script << "plot '" << dataFile << "' using (bin($2,binwidth)):(1.0) smooth freq with boxes lc rgb 'green' title 'Distribution'" << std::endl;
    
    
    script << "set title 'Q-Q Plot (Normality Check)'" << std::endl;
    script << "set xlabel 'Theoretical Quantiles'" << std::endl;
    script << "set ylabel 'Sample Quantiles'" << std::endl;
    script << "set grid" << std::endl;
    script << "stats '" << dataFile << "' using 2 nooutput" << std::endl;
    script << "plot '" << dataFile << "' using (invnorm(($0+1)/(STATS_records+1))):2 with points pt 7 lc rgb 'purple' title 'Q-Q points', \\" << std::endl;
    script << "     x with lines lw 2 lc rgb 'red' title 'Normal line'" << std::endl;
    
    
    script << "set title 'Autocorrelation Function'" << std::endl;
    script << "set xlabel 'Lag'" << std::endl;
    script << "set ylabel 'Autocorrelation'" << std::endl;
    script << "set grid" << std::endl;
    script << "set yrange [-1:1]" << std::endl;
    script << "plot '" << dataFile << "' using 1:2 with lines lw 1 lc rgb 'orange' title 'ACF'" << std::endl;
    
    script << "unset multiplot" << std::endl;
    script.close();
    
    std::string command = "gnuplot \"" + scriptFile + "\"";
    int result = std::system(command.c_str());
    
    if (result == 0) {
        std::cout << "Residual analysis plot saved: plots/residuals/residuals_analysis.png" << std::endl;
    }
    
    
    cleanupDataFiles({dataFile, scriptFile});
}





void SeriesControl::forecastAnalysis(size_t shortStep, size_t longStep) const {
    std::cout << "=== FORECAST ANALYSIS ===" << std::endl;
    std::cout << "Series: " << series.getName() << std::endl;
    std::cout << "Steps: " << shortStep << " and " << longStep << std::endl;
    
    size_t max_steps = std::max(shortStep, longStep);
    
    
    std::cout << "\nLINEAR TREND FORECAST:" << std::endl;
    auto linear_forecasts = series.forecastLinear(max_steps);
    for (size_t step : {shortStep, longStep}) {
        if (step <= linear_forecasts.size()) {
            auto& f = linear_forecasts[step - 1];
            double span = f.upper - f.lower;
            std::cout << "Step " << step << ": Expected = " << f.point 
                      << ", Confidence range = [" << f.lower << ", " << f.upper << "]"
                      << ", Total interval span = " << span << std::endl;
        }
    }
    
    
    std::cout << "\nPOLYNOMIAL2 TREND FORECAST:" << std::endl;
    auto poly2_forecasts = series.forecastPolynomial2(max_steps);
    for (size_t step : {shortStep, longStep}) {
        if (step <= poly2_forecasts.size()) {
            auto& f = poly2_forecasts[step - 1];
            double span = f.upper - f.lower;
            std::cout << "Step " << step << ": Expected = " << f.point 
                      << ", Confidence range = [" << f.lower << ", " << f.upper << "]"
                      << ", Total interval span = " << span << std::endl;
        }
    }
    
    
    std::cout << "\nPOLYNOMIAL3 TREND FORECAST:" << std::endl;
    auto poly3_forecasts = series.forecastPolynomial3(max_steps);
    for (size_t step : {shortStep, longStep}) {
        if (step <= poly3_forecasts.size()) {
            auto& f = poly3_forecasts[step - 1];
            double span = f.upper - f.lower;
            std::cout << "Step " << step << ": Expected = " << f.point 
                      << ", Confidence range = [" << f.lower << ", " << f.upper << "]"
                      << ", Total interval span = " << span << std::endl;
        }
    }

}

void SeriesControl::forecastAnalysisBrown(double alpha, double train_fraction) {
    std::cout << "=== BROWN'S MODEL FORECAST ANALYSIS ===" << std::endl;
    std::cout << "Series: " << series.getName() << std::endl;
    std::cout << "Alpha: " << alpha << ", Train fraction: " << train_fraction << std::endl;

    auto result = series.backtestExponential(alpha, train_fraction);

    if (result.fixed_points.empty()) {
        std::cout << "Error: Insufficient data for backtesting." << std::endl;
        return;
    }

    // Вывод интервального прогноза для последней точки
    std::cout << "\nLAST POINT INTERVAL FORECAST (Fixed model):" << std::endl;
    std::cout << "Point: " << result.last_fixed_interval.point
              << ", Interval: [" << result.last_fixed_interval.lower << ", " << result.last_fixed_interval.upper << "]"
              << ", Span: " << (result.last_fixed_interval.upper - result.last_fixed_interval.lower) << std::endl;

    std::cout << "\nLAST POINT INTERVAL FORECAST (Adaptive model):" << std::endl;
    std::cout << "Point: " << result.last_adaptive_interval.point
              << ", Interval: [" << result.last_adaptive_interval.lower << ", " << result.last_adaptive_interval.upper << "]"
              << ", Span: " << (result.last_adaptive_interval.upper - result.last_adaptive_interval.lower) << std::endl;

    // MSE сравнение
    std::cout << "\nMSE (Fixed): " << result.mse_fixed << std::endl;
    std::cout << "MSE (Adaptive): " << result.mse_adapt << std::endl;

    // Мотивированный вывод
    std::cout << "\n=== QUALITY CONCLUSION ===" << std::endl;
    if (result.mse_adapt < result.mse_fixed) {
        std::cout << "Adaptive model is better (lower MSE: " << result.mse_adapt << " vs " << result.mse_fixed << "). "
                  << "It adapts to new data, reducing error in dynamic series." << std::endl;
    } else {
        std::cout << "Fixed model is better or comparable (MSE: " << result.mse_fixed << " vs " << result.mse_adapt << "). "
                  << "Series may be stable, no need for continuous adaptation." << std::endl;
    }
    double fixed_span = result.last_fixed_interval.upper - result.last_fixed_interval.lower;
    double adapt_span = result.last_adaptive_interval.upper - result.last_adaptive_interval.lower;
    if (adapt_span < fixed_span) {
        std::cout << "Adaptive intervals are narrower (" << adapt_span << " vs " << fixed_span << "), indicating higher confidence." << std::endl;
    } else {
        std::cout << "Fixed intervals are narrower or similar (" << fixed_span << " vs " << adapt_span << "). Adaptive may overfit." << std::endl;
    }
    std::cout << "Overall: Brown's model suits short-term forecasts; for long-term, consider trends if MSE is high." << std::endl;

    // Построение графика
    plotForecastBrown(result);
}

void SeriesControl::plotForecastBrown(const SeriesClass::ForecastBacktestResult& result) const {
    std::cout << "=== GENERATING BROWN'S FORECAST PLOT ===" << std::endl;

    std::filesystem::create_directories("plots/forecast");

    size_t n = series.size();
    size_t split = n - result.fixed_points.size();  // Вычисляем split обратно

    // Сохранение данных: original, fixed, adaptive
    std::string originalFile = "plots/forecast/brown_original.dat";
    saveDataToFile(series, originalFile);

    std::string fixedFile = "plots/forecast/brown_fixed.dat";
    std::ofstream fixedOut(fixedFile);
    fixedOut << "# Index Value\n";
    for (size_t i = split; i < n; ++i) {
        fixedOut << i << " " << result.fixed_points[i - split] << "\n";
    }
    fixedOut.close();

    std::string adaptiveFile = "plots/forecast/brown_adaptive.dat";
    std::ofstream adaptOut(adaptiveFile);
    adaptOut << "# Index Value\n";
    for (size_t i = split; i < n; ++i) {
        adaptOut << i << " " << result.adaptive_points[i - split] << "\n";
    }
    adaptOut.close();

    // GNUplot скрипт
    std::string scriptFile = "plots/forecast/brown_forecast.gnu";
    std::ofstream script(scriptFile);
    script << "set terminal pngcairo size 1600,800 enhanced font 'Arial,12'\n";
    script << "set output 'plots/forecast/brown_forecast_comparison.png'\n";
    script << "set title \"Brown's Model Forecast: " << series.getName() << "\"\n";    script << "set xlabel 'Time Index'\n";
    script << "set ylabel 'Value'\n";
    script << "set grid\n";
    script << "set key outside\n";
    script << "plot '" << originalFile << "' using 1:3 with lines lw 2 title 'Original', \\\n";
    script << "     '" << fixedFile << "' using 1:2 with lines lw 2 lc rgb 'red' title 'Fixed (learning stopped)', \\\n";
    script << "     '" << adaptiveFile << "' using 1:2 with lines lw 2 lc rgb 'green' title 'Adaptive (continuous learning)'\n";
    script.close();

    std::string command = "gnuplot \"" + scriptFile + "\"";
    int res = std::system(command.c_str());
    if (res == 0) {
        std::cout << "Plot saved: plots/forecast/brown_forecast_comparison.png" << std::endl;
    } else {
        std::cerr << "GNUplot error." << std::endl;
    }

    // Cleanup
    cleanupDataFiles({originalFile, fixedFile, adaptiveFile, scriptFile});
}