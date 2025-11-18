#include "SeriesClass.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <sstream>

SeriesClass::SeriesClass(const std::vector<double>& values, 
                       const std::vector<std::string>& times, 
                       const std::string& seriesName)
    : data(values), timestamps(times), name(seriesName) {
    if (values.size() != times.size()) {
        throw std::invalid_argument("Data and timestamps must have the same size");
    }
}

const std::vector<double>& SeriesClass::getData() const { return data; }
const std::vector<std::string>& SeriesClass::getTimestamps() const { return timestamps; }
std::string SeriesClass::getName() const { return name; }
size_t SeriesClass::size() const { return data.size(); }

void SeriesClass::replaceData(const std::vector<double>& newData) {
    if (newData.size() != data.size()) {
        throw std::invalid_argument("New data must have the same size as original data");
    }
    data = newData;
}

std::vector<size_t> SeriesClass::detectAnomaliesIrwin(double criticalValue) const {
    std::vector<size_t> anomalies;
    if (data.size() < 2) return anomalies;
    
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double sumSq = 0.0;
    for (double value : data) {
        sumSq += (value - mean) * (value - mean);
    }
    double stdDev = std::sqrt(sumSq / (data.size() - 1));
    
    if (stdDev == 0) return anomalies;
    
    for (size_t i = 1; i < data.size(); ++i) {
        double difference = std::abs(data[i] - data[i-1]);
        double lambda = difference / stdDev;
        if (lambda > criticalValue) {
            anomalies.push_back(i);
        }
    }
    
    return anomalies;
}

void SeriesClass::interpolateAnomalies(const std::vector<size_t>& anomalyIndices) {
    for (size_t idx : anomalyIndices) {
        double originalValue = data[idx];
        if (idx == 0) {
            if (data.size() > 1) {
                data[idx] = data[idx + 1];
            }
        } else if (idx == data.size() - 1) {
            data[idx] = data[idx - 1];
        } else {
            data[idx] = (data[idx - 1] + data[idx + 1]) / 2.0;
        }
    }
}

std::vector<double> SeriesClass::movingAverage(size_t window) const {
    std::vector<double> result;
    if (data.empty() || window == 0 || window > data.size()) return result;
    
    result.reserve(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        size_t start = (i < window/2) ? 0 : i - window/2;
        size_t end = std::min(start + window, data.size());
        double sum = 0.0;
        for (size_t j = start; j < end; ++j) {
            sum += data[j];
        }
        result.push_back(sum / (end - start));
    }
    return result;
}

std::vector<double> SeriesClass::weightedMovingAverage(const std::vector<double>& weights) const {
    std::vector<double> result;
    if (data.empty() || weights.empty()) return result;
    
    size_t window = weights.size();
    if (window > data.size()) return result;
    
    result.reserve(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        if (i < window/2 || i >= data.size() - window/2) {
            result.push_back(data[i]);
            continue;
        }
        double weightedSum = 0.0;
        double weightSum = 0.0;
        for (size_t j = 0; j < window; ++j) {
            int dataIdx = static_cast<int>(i) - static_cast<int>(window/2) + static_cast<int>(j);
            if (dataIdx >= 0 && dataIdx < static_cast<int>(data.size())) {
                weightedSum += data[dataIdx] * weights[j];
                weightSum += weights[j];
            }
        }
        result.push_back(weightedSum / weightSum);
    }
    return result;
}

std::vector<double> SeriesClass::exponentialSmoothing(double alpha) const {
    std::vector<double> result;
    if (data.empty()) return result;
    
    result.reserve(data.size());
    result.push_back(data[0]);
    for (size_t i = 1; i < data.size(); ++i) {
        double smoothed = alpha * data[i] + (1 - alpha) * result[i-1];
        result.push_back(smoothed);
    }
    return result;
}

std::vector<double> SeriesClass::generateLinearWeights(size_t n) {
    std::vector<double> weights(n);
    double total = n * (n + 1) / 2.0;
    for (size_t i = 0; i < n; ++i) {
        weights[i] = (i + 1) / total;
    }
    return weights;
}

std::vector<double> SeriesClass::generateTriangularWeights(size_t n) {
    std::vector<double> weights(n);
    if (n % 2 == 0) {
        size_t half = n / 2;
        for (size_t i = 0; i < half; ++i) {
            weights[i] = i + 1;
            weights[n - 1 - i] = i + 1;
        }
    } else {
        size_t center = n / 2;
        for (size_t i = 0; i <= center; ++i) {
            weights[i] = i + 1;
            weights[n - 1 - i] = i + 1;
        }
    }
    double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    for (double& weight : weights) {
        weight /= sum;
    }
    return weights;
}

std::pair<bool, double> SeriesClass::checkTrendMeanDifferences() const {
    if (data.size() < 3) return {false, 0.0};

    size_t n = data.size();
    size_t splitPoint = n / 2;
    size_t n1 = splitPoint;
    size_t n2 = n - splitPoint;

    double mean1 = 0.0, mean2 = 0.0;
    for (size_t i = 0; i < n1; ++i) mean1 += data[i];
    mean1 /= n1;
    for (size_t i = splitPoint; i < n; ++i) mean2 += data[i];
    mean2 /= n2;

    double var1 = 0.0, var2 = 0.0;
    for (size_t i = 0; i < n1; ++i) var1 += (data[i] - mean1) * (data[i] - mean1);
    var1 /= (n1 - 1);
    for (size_t i = splitPoint; i < n; ++i) var2 += (data[i] - mean2) * (data[i] - mean2);
    var2 /= (n2 - 1);

    double t_statistic = (mean1 - mean2) / std::sqrt(var1 / n1 + var2 / n2);
    double F_statistic = (var1 > var2) ? var1 / var2 : var2 / var1;
    double df1 = n1 - 1.0;
    double df2 = n2 - 1.0;
    double z = 1.96;
    double f_critical = std::exp(z * std::sqrt(2.0 * (1.0 / df1 + 1.0 / df2)));
    double t_critical = 1.96;
    bool has_trend = std::abs(t_statistic) > t_critical;

    return {has_trend, t_statistic};
}

std::pair<bool, double> SeriesClass::checkTrendFosterStewart() const {
    if (data.size() < 3) return {false, 0.0};

    size_t n = data.size();
    double m_i = data[0];
    double M_i = data[0];
    int u = 0, d = 0;

    for (size_t i = 1; i < n; ++i) {
        if (data[i] > M_i) {
            ++u;
            M_i = data[i];
        } else if (data[i] < m_i) {
            ++d;
            m_i = data[i];
        }
    }

    double S = static_cast<double>(u + d);
    double D = static_cast<double>(u - d);
    double E_S = 0.0;
    double Var_S = 0.0;
    for (size_t i = 1; i <= n; ++i) {
        E_S += 2.0 / i;
        Var_S += 4.0 / i - 8.0 / (i * i);
    }
    E_S -= 1.0;
    Var_S = std::max(Var_S, 1e-9);

    double ts = (S - E_S) / std::sqrt(Var_S);
    double td = D / std::sqrt(Var_S);
    double critical_value = 1.96;
    bool has_trend = (std::abs(ts) > critical_value) || (std::abs(td) > critical_value);

    return {has_trend, std::max(std::abs(ts), std::abs(td))};
}

SeriesClass::DecompositionResult SeriesClass::decomposeTimeSeries(int period) const {
    DecompositionResult result;
    size_t n = data.size();
    if (n < period * 2) {
        throw std::invalid_argument("Series too short for decomposition");
    }
    
    result.original = data;
    result.preliminaryTrend.resize(n, 0.0);
    result.primaryTrend.resize(n, 0.0);
    result.secondaryTrend.resize(n, 0.0);
    result.finalTrend.resize(n, 0.0);
    result.preliminarySeasonal.resize(n, 0.0);
    result.primarySeasonal.resize(n, 0.0);
    result.secondarySeasonal.resize(n, 0.0);
    result.finalSeasonal.resize(n, 0.0);
    result.preliminaryResidual.resize(n, 0.0);
    result.primaryResidual.resize(n, 0.0);
    result.secondaryResidual.resize(n, 0.0);
    result.finalResidual.resize(n, 0.0);
        
    int window1 = 30;
    for (size_t i = 0; i < n; ++i) {
        size_t start = (i < window1/2) ? 0 : i - window1/2;
        size_t end = std::min(start + window1, n);
        double sum = 0.0;
        size_t count = 0;
        for (size_t j = start; j < end; ++j) {
            sum += data[j];
            count++;
        }
        result.preliminaryTrend[i] = (count > 0) ? sum / count : data[i];
    }
    
    auto weights7 = generateLinearWeights(7);
    auto wma7Data = weightedMovingAverage(weights7);
    if (wma7Data.size() == n) {
        result.primaryTrend = wma7Data;
    } else {
        result.primaryTrend = result.preliminaryTrend;
    }
    
    int trendWindow = period;
    for (size_t i = 0; i < n; ++i) {
        int halfWindow = trendWindow / 2;
        int start = static_cast<int>(i) - halfWindow;
        int end = static_cast<int>(i) + halfWindow + (trendWindow % 2);
        if (start < 0) start = 0;
        if (end > static_cast<int>(n)) end = n;
        double sum = 0.0;
        int count = 0;
        for (int j = start; j < end; ++j) {
            sum += data[j];
            count++;
        }
        result.finalTrend[i] = (count > 0) ? sum / count : data[i];
    }
    result.secondaryTrend = result.finalTrend;
    
    std::vector<double> detrended(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        detrended[i] = data[i] - result.finalTrend[i];
    }
    
    std::vector<std::vector<double>> seasonalGroups(period);
    for (size_t i = 0; i < n; ++i) {
        int seasonIndex = i % period;
        seasonalGroups[seasonIndex].push_back(detrended[i]);
    }
    
    std::vector<double> seasonalPattern(period, 0.0);
    for (int i = 0; i < period; ++i) {
        if (!seasonalGroups[i].empty()) {
            std::vector<double> sorted = seasonalGroups[i];
            std::sort(sorted.begin(), sorted.end());
            size_t mid = sorted.size() / 2;
            if (sorted.size() % 2 == 0) {
                seasonalPattern[i] = (sorted[mid-1] + sorted[mid]) / 2.0;
            } else {
                seasonalPattern[i] = sorted[mid];
            }
        }
    }
    
    double seasonalSum = std::accumulate(seasonalPattern.begin(), seasonalPattern.end(), 0.0);
    double adjustment = seasonalSum / period;
    for (int i = 0; i < period; ++i) {
        seasonalPattern[i] -= adjustment;
    }
    
    for (size_t i = 0; i < n; ++i) {
        int seasonIndex = i % period;
        result.finalSeasonal[i] = seasonalPattern[seasonIndex];
        result.secondarySeasonal[i] = result.finalSeasonal[i];
    }
        
    for (size_t i = 0; i < n; ++i) {
        result.finalResidual[i] = data[i] - result.finalTrend[i] - result.finalSeasonal[i];
        result.secondaryResidual[i] = result.finalResidual[i];
        result.preliminarySeasonal[i] = data[i] - result.preliminaryTrend[i];
        result.primarySeasonal[i] = data[i] - result.primaryTrend[i];
        result.preliminaryResidual[i] = data[i] - (result.preliminaryTrend[i] + result.preliminarySeasonal[i]);
        result.primaryResidual[i] = data[i] - (result.primaryTrend[i] + result.primarySeasonal[i]);
    }
    
    return result;
}

std::vector<double> SeriesClass::predictLogistic(double k, double A) const {
    std::vector<double> predictions;
    predictions.reserve(data.size());
    double L = *std::max_element(data.begin(), data.end()) * 1.2;
    for (size_t t = 0; t < data.size(); ++t) {
        double prediction = L / (1.0 + A * std::exp(-k * static_cast<double>(t)));
        predictions.push_back(prediction);
    }
    return predictions;
}

std::vector<double> SeriesClass::predictGompertz(double k, double t0) const {
    std::vector<double> predictions;
    predictions.reserve(data.size());
    double L = *std::max_element(data.begin(), data.end()) * 1.1;
    for (size_t t = 0; t < data.size(); ++t) {
        double prediction = L * std::exp(-std::exp(-k * (static_cast<double>(t) - t0)));
        predictions.push_back(prediction);
    }
    return predictions;
}

std::vector<double> SeriesClass::smoothThreePoints() const {
    std::vector<double> smoothed;
    if (data.empty()) return smoothed;
    smoothed.reserve(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        if (i == 0) {
            smoothed.push_back((data[0] + data[1]) / 2.0);
        } else if (i == data.size() - 1) {
            smoothed.push_back((data[data.size()-2] + data[data.size()-1]) / 2.0);
        } else {
            smoothed.push_back((data[i-1] + data[i] + data[i+1]) / 3.0);
        }
    }
    return smoothed;
}

std::pair<double, double> SeriesClass::fitLogistic() const {
    if (data.empty() || data.size() < 10) return {0.0, 0.0};
    double max_val = *std::max_element(data.begin(), data.end());
    double min_val = *std::min_element(data.begin(), data.end());
    if (max_val <= min_val) return {0.0, 0.0};
    double L = max_val * 1.2;
    std::vector<double> z_values;
    std::vector<double> t_values;
    for (size_t t = 0; t < data.size(); ++t) {
        if (data[t] > 0 && data[t] < L * 0.99) {
            double ratio = (L - data[t]) / data[t];
            if (ratio > 1e-10) {
                double z = std::log(ratio);
                if (std::isfinite(z)) {
                    z_values.push_back(z);
                    t_values.push_back(static_cast<double>(t));
                }
            }
        }
    }
    if (z_values.size() < 5) return {0.001, 1.0};
    double sum_t = 0.0, sum_z = 0.0, sum_tz = 0.0, sum_t2 = 0.0;
    size_t n = z_values.size();
    for (size_t i = 0; i < n; ++i) {
        sum_t += t_values[i];
        sum_z += z_values[i];
        sum_tz += t_values[i] * z_values[i];
        sum_t2 += t_values[i] * t_values[i];
    }
    double denominator = n * sum_t2 - sum_t * sum_t;
    if (std::abs(denominator) < 1e-10) return {0.001, 1.0};
    double k = -(n * sum_tz - sum_t * sum_z) / denominator;
    double ln_A = (sum_z + k * sum_t) / n;
    double A = std::exp(ln_A);
    k = std::max(0.0001, std::min(k, 1.0));
    A = std::max(0.1, std::min(A, 100.0));
    return {k, A};
}

std::pair<double, double> SeriesClass::fitGompertz() const {
    if (data.empty() || data.size() < 10) return {0.0, 0.0};
    double max_val = *std::max_element(data.begin(), data.end());
    double min_val = *std::min_element(data.begin(), data.end());
    if (max_val <= min_val) return {0.0, 0.0};
    double L = max_val * 1.1;
    std::vector<double> z_values;
    std::vector<double> t_values;
    for (size_t t = 0; t < data.size(); ++t) {
        if (data[t] > 0 && data[t] < L * 0.99) {
            double ratio = L / data[t];
            if (ratio > 1.0) {
                double z = std::log(std::log(ratio));
                if (std::isfinite(z)) {
                    z_values.push_back(z);
                    t_values.push_back(static_cast<double>(t));
                }
            }
        }
    }
    if (z_values.size() < 5) return {0.001, 0.0};
    double sum_t = 0.0, sum_z = 0.0, sum_tz = 0.0, sum_t2 = 0.0;
    size_t n = z_values.size();
    for (size_t i = 0; i < n; ++i) {
        sum_t += t_values[i];
        sum_z += z_values[i];
        sum_tz += t_values[i] * z_values[i];
        sum_t2 += t_values[i] * t_values[i];
    }
    double denominator = n * sum_t2 - sum_t * sum_t;
    if (std::abs(denominator) < 1e-10) return {0.001, 0.0};
    double b = (n * sum_tz - sum_t * sum_z) / denominator;
    double a = (sum_z - b * sum_t) / n;
    double k = -b;
    double t0 = a / b;
    k = std::max(0.0001, std::min(k, 1.0));
    return {k, t0};
}

std::vector<double> SeriesClass::calculateFirstDifferences() const {
    std::vector<double> differences;
    if (data.size() < 2) return differences;
    differences.reserve(data.size() - 1);
    for (size_t i = 1; i < data.size(); ++i) {
        differences.push_back(data[i] - data[i-1]);
    }
    return differences;
}

std::vector<double> SeriesClass::calculateSecondDifferences() const {
    auto firstDiff = calculateFirstDifferences();
    std::vector<double> secondDiff;
    if (firstDiff.size() < 2) return secondDiff;
    secondDiff.reserve(firstDiff.size() - 1);
    for (size_t i = 1; i < firstDiff.size(); ++i) {
        secondDiff.push_back(firstDiff[i] - firstDiff[i-1]);
    }
    return secondDiff;
}

std::vector<double> SeriesClass::calculateRelativeFirstDifferences() const {
    std::vector<double> relative;
    if (data.size() < 2) return relative;
    relative.reserve(data.size() - 1);
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i-1] != 0) {
            relative.push_back((data[i] - data[i-1]) / data[i-1]);
        } else {
            relative.push_back(0.0);
        }
    }
    return relative;
}

std::vector<double> SeriesClass::calculateLogFirstDifferences() const {
    auto firstDiff = calculateFirstDifferences();
    std::vector<double> logDiff;
    for (double diff : firstDiff) {
        if (diff > 0) {
            logDiff.push_back(std::log(diff));
        } else {
            logDiff.push_back(0.0);
        }
    }
    return logDiff;
}

std::vector<double> SeriesClass::calculateGompertzIndicator() const {
    auto relativeDiff = calculateRelativeFirstDifferences();
    std::vector<double> gompertz;
    for (size_t i = 0; i < relativeDiff.size() && i < data.size() - 1; ++i) {
        if (data[i] > 0 && std::abs(data[i]) > 1e-10 && 
            std::abs(relativeDiff[i]) > 1e-10 && relativeDiff[i] > 0) {
            double ratio = relativeDiff[i] / data[i];
            if (ratio > 1e-10) {
                gompertz.push_back(std::log(ratio));
            } else {
                gompertz.push_back(0.0);
            }
        } else {
            gompertz.push_back(0.0);
        }
    }
    return gompertz;
}

std::vector<double> SeriesClass::calculateLogisticIndicator() const {
    auto relativeDiff = calculateRelativeFirstDifferences();
    std::vector<double> logistic;
    for (size_t i = 0; i < relativeDiff.size() && i < data.size() - 1; ++i) {
        if (data[i] > 0 && std::abs(data[i]) > 1e-10 && 
            data[i] * data[i] > 1e-10 &&  
            std::abs(relativeDiff[i]) > 1e-10 && relativeDiff[i] > 0) {
            double term = relativeDiff[i] / (data[i] * data[i]);
            if (term > 1e-10) {
                logistic.push_back(std::log(term));
            } else {
                logistic.push_back(0.0);
            }
        } else {
            logistic.push_back(0.0);
        }
    }
    return logistic;
}

std::pair<double, double> SeriesClass::fitLinearPolynomial() const {
    if (data.empty()) return {0.0, 0.0};
    double sum_t = 0.0, sum_y = 0.0, sum_ty = 0.0, sum_t2 = 0.0;
    size_t n = data.size();
    for (size_t t = 0; t < n; ++t) {
        sum_t += t;
        sum_y += data[t];
        sum_ty += t * data[t];
        sum_t2 += t * t;
    }
    double denominator = n * sum_t2 - sum_t * sum_t;
    if (denominator == 0) return {0.0, 0.0};
    double b = (n * sum_ty - sum_t * sum_y) / denominator;
    double a = (sum_y - b * sum_t) / n;
    return {a, b};
}

std::pair<double, double> SeriesClass::fitExponential() const {
    if (data.empty()) return {0.0, 0.0};
    double sum_t = 0.0, sum_ln_y = 0.0, sum_t_ln_y = 0.0, sum_t2 = 0.0;
    size_t n = data.size();
    for (size_t t = 0; t < n; ++t) {
        if (data[t] > 0) {
            sum_t += t;
            sum_ln_y += std::log(data[t]);
            sum_t_ln_y += t * std::log(data[t]);
            sum_t2 += t * t;
        }
    }
    double denominator = n * sum_t2 - sum_t * sum_t;
    if (denominator == 0) return {0.0, 0.0};
    double b = (n * sum_t_ln_y - sum_t * sum_ln_y) / denominator;
    double ln_a = (sum_ln_y - b * sum_t) / n;
    return {std::exp(ln_a), b};
}

std::vector<double> SeriesClass::predictLinear(double a, double b) const {
    std::vector<double> predictions;
    predictions.reserve(data.size());
    for (size_t t = 0; t < data.size(); ++t) {
        predictions.push_back(a + b * t);
    }
    return predictions;
}

std::vector<double> SeriesClass::predictExponential(double a, double b) const {
    std::vector<double> predictions;
    predictions.reserve(data.size());
    for (size_t t = 0; t < data.size(); ++t) {
        predictions.push_back(a * std::exp(b * t));
    }
    return predictions;
}

void SeriesClass::analyzeGrowthCurveCharacteristics() const {
    auto firstDiff = calculateFirstDifferences();
    auto secondDiff = calculateSecondDifferences();
    auto relativeDiff = calculateRelativeFirstDifferences();
    auto logFirstDiff = calculateLogFirstDifferences();
    auto gompertzInd = calculateGompertzIndicator();
    auto logisticInd = calculateLogisticIndicator();
    
    int valid_gompertz = 0, valid_logistic = 0;
    double sum_gompertz = 0.0, sum_logistic = 0.0;
    
    for (size_t i = 0; i < gompertzInd.size(); ++i) {
        if (std::isfinite(gompertzInd[i]) && std::abs(gompertzInd[i]) > 1e-10) {
            sum_gompertz += gompertzInd[i];
            valid_gompertz++;
        }
    }
    
    for (size_t i = 0; i < logisticInd.size(); ++i) {
        if (std::isfinite(logisticInd[i]) && std::abs(logisticInd[i]) > 1e-10) {
            sum_logistic += logisticInd[i];
            valid_logistic++;
        }
    }
    
    double meanFirstDiff = firstDiff.empty() ? 0.0 : 
        std::accumulate(firstDiff.begin(), firstDiff.end(), 0.0) / firstDiff.size();
    double meanSecondDiff = secondDiff.empty() ? 0.0 :
        std::accumulate(secondDiff.begin(), secondDiff.end(), 0.0) / secondDiff.size();
    double meanRelativeDiff = relativeDiff.empty() ? 0.0 :
        std::accumulate(relativeDiff.begin(), relativeDiff.end(), 0.0) / relativeDiff.size();
    
    double meanLogFirstDiff = 0.0;
    int count_log = 0;
    for (double val : logFirstDiff) {
        if (val != 0.0 && std::isfinite(val)) {
            meanLogFirstDiff += val;
            count_log++;
        }
    }
    meanLogFirstDiff = (count_log > 0) ? meanLogFirstDiff / count_log : 0.0;
    
    double meanGompertz = (valid_gompertz > 0) ? sum_gompertz / valid_gompertz : 0.0;
    double meanLogistic = (valid_logistic > 0) ? sum_logistic / valid_logistic : 0.0;
    
    bool constant_first_diff = (std::abs(meanSecondDiff) < 0.1 * std::abs(meanFirstDiff));
    bool gompertz_available = (valid_gompertz > data.size() * 0.1);
    bool logistic_available = (valid_logistic > data.size() * 0.1);
}

std::vector<double> SeriesClass::fitPolynomial2() const {
    if (data.empty() || data.size() < 3) return {0.0, 0.0, 0.0};
    double sum_t = 0.0, sum_t2 = 0.0, sum_t3 = 0.0, sum_t4 = 0.0;
    double sum_y = 0.0, sum_ty = 0.0, sum_t2y = 0.0;
    size_t n = data.size();
    for (size_t t = 0; t < n; ++t) {
        double t_val = static_cast<double>(t);
        double t2 = t_val * t_val;
        double t3 = t2 * t_val;
        double t4 = t3 * t_val;
        sum_t += t_val;
        sum_t2 += t2;
        sum_t3 += t3;
        sum_t4 += t4;
        sum_y += data[t];
        sum_ty += t_val * data[t];
        sum_t2y += t2 * data[t];
    }
    double det = n * (sum_t2 * sum_t4 - sum_t3 * sum_t3) 
               - sum_t * (sum_t * sum_t4 - sum_t2 * sum_t3)
               + sum_t2 * (sum_t * sum_t3 - sum_t2 * sum_t2);
    if (std::abs(det) < 1e-12) {
        double b_linear = (n * sum_ty - sum_t * sum_y) / (n * sum_t2 - sum_t * sum_t);
        double a_linear = (sum_y - b_linear * sum_t) / n;
        return {a_linear, b_linear, 0.0};
    }
    double det_a = sum_y * (sum_t2 * sum_t4 - sum_t3 * sum_t3)
                 - sum_t * (sum_ty * sum_t4 - sum_t2y * sum_t3)
                 + sum_t2 * (sum_ty * sum_t3 - sum_t2y * sum_t2);
    double det_b = n * (sum_ty * sum_t4 - sum_t2y * sum_t3)
                 - sum_y * (sum_t * sum_t4 - sum_t2 * sum_t3)
                 + sum_t2 * (sum_t * sum_t2y - sum_ty * sum_t2);
    double det_c = n * (sum_t2 * sum_t2y - sum_t3 * sum_ty)
                 - sum_t * (sum_t * sum_t2y - sum_t2 * sum_ty)
                 + sum_y * (sum_t * sum_t3 - sum_t2 * sum_t2);
    double a = det_a / det;
    double b = det_b / det;
    double c = det_c / det;
    return {a, b, c};
}

std::vector<double> SeriesClass::fitPolynomial3() const {
    if (data.empty() || data.size() < 4) return {0.0, 0.0, 0.0, 0.0};
    double sum_t = 0.0, sum_t2 = 0.0, sum_t3 = 0.0, sum_t4 = 0.0, sum_t5 = 0.0, sum_t6 = 0.0;
    double sum_y = 0.0, sum_ty = 0.0, sum_t2y = 0.0, sum_t3y = 0.0;
    size_t n = data.size();
    for (size_t t = 0; t < n; ++t) {
        double t_val = static_cast<double>(t);
        double t2 = t_val * t_val;
        double t3 = t2 * t_val;
        double t4 = t3 * t_val;
        double t5 = t4 * t_val;
        double t6 = t5 * t_val;
        sum_t += t_val;
        sum_t2 += t2;
        sum_t3 += t3;
        sum_t4 += t4;
        sum_t5 += t5;
        sum_t6 += t6;
        sum_y += data[t];
        sum_ty += t_val * data[t];
        sum_t2y += t2 * data[t];
        sum_t3y += t3 * data[t];
    }
    double mat[4][4] = {
        {static_cast<double>(n), sum_t, sum_t2, sum_t3},
        {sum_t, sum_t2, sum_t3, sum_t4},
        {sum_t2, sum_t3, sum_t4, sum_t5},
        {sum_t3, sum_t4, sum_t5, sum_t6}
    };
    double vec[4] = {sum_y, sum_ty, sum_t2y, sum_t3y};
    for (int i = 0; i < 4; ++i) {
        int max_row = i;
        for (int k = i + 1; k < 4; ++k) {
            if (std::abs(mat[k][i]) > std::abs(mat[max_row][i])) {
                max_row = k;
            }
        }
        for (int k = i; k < 4; ++k) {
            std::swap(mat[i][k], mat[max_row][k]);
        }
        std::swap(vec[i], vec[max_row]);
        if (std::abs(mat[i][i]) < 1e-12) {
            return {0.0, 0.0, 0.0, 0.0};
        }
        for (int k = i + 1; k < 4; ++k) {
            double factor = mat[k][i] / mat[i][i];
            for (int j = i; j < 4; ++j) {
                mat[k][j] -= factor * mat[i][j];
            }
            vec[k] -= factor * vec[i];
        }
    }
    double coeffs[4];
    for (int i = 3; i >= 0; --i) {
        coeffs[i] = vec[i];
        for (int j = i + 1; j < 4; ++j) {
            coeffs[i] -= mat[i][j] * coeffs[j];
        }
        coeffs[i] /= mat[i][i];
    }
    return {coeffs[0], coeffs[1], coeffs[2], coeffs[3]};
}

std::vector<double> SeriesClass::predictPolynomial2(const std::vector<double>& coeffs) const {
    std::vector<double> predictions;
    if (coeffs.size() != 3) return predictions;
    predictions.reserve(data.size());
    double a = coeffs[0], b = coeffs[1], c = coeffs[2];
    for (size_t t = 0; t < data.size(); ++t) {
        double t_val = static_cast<double>(t);
        predictions.push_back(a + b * t_val + c * t_val * t_val);
    }
    return predictions;
}

std::vector<double> SeriesClass::predictPolynomial3(const std::vector<double>& coeffs) const {
    std::vector<double> predictions;
    if (coeffs.size() != 4) return predictions;
    predictions.reserve(data.size());
    double a = coeffs[0], b = coeffs[1], c = coeffs[2], d = coeffs[3];
    for (size_t t = 0; t < data.size(); ++t) {
        double t_val = static_cast<double>(t);
        double t2 = t_val * t_val;
        double t3 = t2 * t_val;
        predictions.push_back(a + b * t_val + c * t2 + d * t3);
    }
    return predictions;
}

double SeriesClass::calculateSkewness() const {
    if (data.size() < 3) return 0.0;
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double sum3 = 0.0, sum2 = 0.0;
    for (double value : data) {
        double diff = value - mean;
        sum3 += diff * diff * diff;
        sum2 += diff * diff;
    }
    double variance = sum2 / (data.size() - 1);
    if (variance == 0) return 0.0;
    return (sum3 / data.size()) / std::pow(variance, 1.5);
}

double SeriesClass::calculateKurtosis() const {
    if (data.size() < 4) return 0.0;
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double sum4 = 0.0, sum2 = 0.0;
    for (double value : data) {
        double diff = value - mean;
        sum4 += diff * diff * diff * diff;
        sum2 += diff * diff;
    }
    double variance = sum2 / (data.size() - 1);
    if (variance == 0) return 0.0;
    return (sum4 / data.size()) / (variance * variance) - 3.0;
}

double SeriesClass::calculateRSStatistic() const {
    if (data.size() < 2) return 0.0;
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double max = *std::max_element(data.begin(), data.end());
    double min = *std::min_element(data.begin(), data.end());
    double sumSq = 0.0;
    for (double value : data) {
        sumSq += (value - mean) * (value - mean);
    }
    double stdDev = std::sqrt(sumSq / (data.size() - 1));
    if (stdDev == 0) return 0.0;
    return (max - min) / stdDev;
}

double SeriesClass::calculateDurbinWatson() const {
    if (data.size() < 3) return 0.0;
    double numerator = 0.0;
    for (size_t i = 1; i < data.size(); ++i) {
        numerator += (data[i] - data[i-1]) * (data[i] - data[i-1]);
    }
    double denominator = 0.0;
    for (double value : data) {
        denominator += value * value;
    }
    if (denominator == 0) return 0.0;
    return numerator / denominator;
}

std::pair<size_t, bool> SeriesClass::analyzeTurningPoints() const {
    if (data.size() < 3) return {0, 0};
    size_t turningPoints = 0;
    for (size_t i = 1; i < data.size() - 1; ++i) {
        if ((data[i] > data[i-1] && data[i] > data[i+1]) || 
            (data[i] < data[i-1] && data[i] < data[i+1])) {
            turningPoints++;
        }
    }
    double expected = 2.0 * (data.size() - 2) / 3.0;
    double variance = (16.0 * data.size() - 29.0) / 90.0;
    double statistic = std::abs(turningPoints - expected) / std::sqrt(variance);
    bool isRandom = statistic < 1.96;
    return { turningPoints, isRandom };
}

std::pair<size_t, bool> SeriesClass::analyzeSeriesTest() const {
    if (data.size() < 2) return {0, 0};
    double median = 0.0;
    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    if (sorted.size() % 2 == 0) {
        median = (sorted[sorted.size()/2 - 1] + sorted[sorted.size()/2]) / 2.0;
    } else {
        median = sorted[sorted.size()/2];
    }
    std::vector<int> signs;
    for (double value : data) {
        signs.push_back(value > median ? 1 : -1);
    }
    size_t seriesCount = 1;
    for (size_t i = 1; i < signs.size(); ++i) {
        if (signs[i] != signs[i-1]) {
            seriesCount++;
        }
    }
    size_t n1 = std::count(signs.begin(), signs.end(), 1);
    size_t n2 = signs.size() - n1;
    double expected = 1.0 + 2.0 * n1 * n2 / (n1 + n2);
    double variance = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n1 - n2) / 
                     ((n1 + n2) * (n1 + n2) * (n1 + n2 - 1));
    double statistic = std::abs(seriesCount - expected) / std::sqrt(variance);
    bool isRandom = statistic < 1.96;
    return {seriesCount, isRandom};
}

SeriesClass::ResidualAnalysisResult SeriesClass::analyzeResiduals() const {
    ResidualAnalysisResult result;
    if (data.empty()) {
        result.conclusion = "No data for analysis";
        return result;
    }
    auto [turningPoints, isRandomTurning] = analyzeTurningPoints();
    result.turningPointsCount = turningPoints;
    result.isRandomByTurningPoints = isRandomTurning;
    auto [seriesCount, isRandomSeries] = analyzeSeriesTest();
    result.seriesCount = seriesCount;
    result.isRandomBySeries = isRandomSeries;
    result.skewness = calculateSkewness();
    result.kurtosis = calculateKurtosis();
    result.isNormalByMoments = (std::abs(result.skewness) < 1.0 && std::abs(result.kurtosis) < 1.0);
    result.rSStatistic = calculateRSStatistic();
    size_t n = data.size();
    bool isNormalRS = (result.rSStatistic > 6.5 && result.rSStatistic < 8.5);
    if (n <= 20) {
        isNormalRS = (result.rSStatistic > 2.0 && result.rSStatistic < 4.0);
    }
    result.isNormalByRS = isNormalRS;
    result.mean = std::accumulate(data.begin(), data.end(), 0.0) / n;
    double sumSq = 0.0;
    for (double value : data) {
        sumSq += (value - result.mean) * (value - result.mean);
    }
    double stdDev = std::sqrt(sumSq / (n - 1));
    if (stdDev > 0) {
        result.tStatistic = std::abs(result.mean) / (stdDev / std::sqrt(n));
        result.hasZeroMean = (result.tStatistic < 2.0);
    } else {
        result.tStatistic = 0.0;
        result.hasZeroMean = true;
    }
    result.durbinWatsonStatistic = calculateDurbinWatson();
    result.isIndependent = (result.durbinWatsonStatistic > 1.5 && 
                           result.durbinWatsonStatistic < 2.5);
    result.isAdequate = result.isRandomByTurningPoints &&
                       result.isRandomBySeries &&
                       result.isNormalByMoments &&
                       result.isNormalByRS &&
                       result.hasZeroMean &&
                       result.isIndependent;
    std::stringstream conclusion;
    conclusion << "=== RESIDUAL ANALYSIS CONCLUSION ===\n";
    conclusion << "Randomness (turning points): " << (result.isRandomByTurningPoints ? "YES" : "NO") << "\n";
    conclusion << "Randomness (series test): " << (result.isRandomBySeries ? "YES" : "NO") << "\n";
    conclusion << "Normality (skewness/kurtosis): " << (result.isNormalByMoments ? "YES" : "NO") << "\n";
    conclusion << "Normality (RS-test): " << (result.isNormalByRS ? "YES" : "NO") << "\n";
    conclusion << "Zero mean (t-test): " << (result.hasZeroMean ? "YES" : "NO") << "\n";
    conclusion << "Independence (Durbin-Watson): " << (result.isIndependent ? "YES" : "NO") << "\n";
    conclusion << "OVERALL MODEL ADEQUACY: " << (result.isAdequate ? "ADEQUATE" : "NOT ADEQUATE") << "\n";
    result.conclusion = conclusion.str();
    return result;
}

double SeriesClass::getTCritical(double alpha, size_t df) const {
    if (df < 2) return 0.0;
    return 1.96;
}

std::vector<SeriesClass::Forecast> SeriesClass::forecastLinear(size_t steps, double alpha) const {
    std::vector<Forecast> results;
    if (data.empty()) return results;
    size_t n = data.size();
    auto [a, b] = fitLinearPolynomial();
    double sum_t = 0.0, sum_t2 = 0.0, sum_y = 0.0, sum_ty = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i);
        double y = data[i];
        sum_t += t;
        sum_t2 += t * t;
        sum_y += y;
        sum_ty += t * y;
    }
    double mean_t = sum_t / n;
    double Sxx = sum_t2 - (sum_t * sum_t) / n;
    double SSE = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i);
        double y_hat = a + b * t;
        SSE += (data[i] - y_hat) * (data[i] - y_hat);
    }
    double MSE = SSE / (n - 2);
    double std_err = std::sqrt(MSE);
    double t_crit = getTCritical(alpha, n - 2);
    for (size_t k = 1; k <= steps; ++k) {
        double t_future = static_cast<double>(n + k - 1);
        double point = a + b * t_future;
        double se_pred = std_err * std::sqrt(1.0 + 1.0 / n + (t_future - mean_t) * (t_future - mean_t) / Sxx);
        double half_width = t_crit * se_pred;
        double lower = point - half_width;
        double upper = point + half_width;
        results.push_back({point, lower, upper});
    }
    return results;
}

std::vector<SeriesClass::Forecast> SeriesClass::forecastPolynomial2(size_t steps, double alpha) const {
    std::vector<Forecast> results;
    if (data.size() < 3) return results;
    size_t n = data.size();
    auto coeffs = fitPolynomial2();
    if (coeffs.size() != 3) return results;
    double a = coeffs[0], b = coeffs[1], c = coeffs[2];
    double SSE = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i);
        double t2 = t * t;
        double y_hat = a + b * t + c * t2;
        SSE += (data[i] - y_hat) * (data[i] - y_hat);
    }
    double MSE = SSE / (n - 3);
    double std_err = std::sqrt(MSE);
    double sum_t = 0.0, sum_t2 = 0.0, sum_t3 = 0.0, sum_t4 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i);
        double t2 = t * t;
        double t3 = t2 * t;
        double t4 = t3 * t;
        sum_t += t;
        sum_t2 += t2;
        sum_t3 += t3;
        sum_t4 += t4;
    }
    double XtX[3][3] = {
        {static_cast<double>(n), sum_t, sum_t2},
        {sum_t, sum_t2, sum_t3},
        {sum_t2, sum_t3, sum_t4}
    };
    double mat[3][6];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            mat[i][j] = XtX[i][j];
            mat[i][j+3] = (i == j) ? 1.0 : 0.0;
        }
    }
    for (int i = 0; i < 3; ++i) {
        if (std::abs(mat[i][i]) < 1e-12) {
            return results;
        }
        double pivot = mat[i][i];
        for (int j = 0; j < 6; ++j) {
            mat[i][j] /= pivot;
        }
        for (int k = 0; k < 3; ++k) {
            if (k != i) {
                double factor = mat[k][i];
                for (int j = 0; j < 6; ++j) {
                    mat[k][j] -= factor * mat[i][j];
                }
            }
        }
    }
    double invXtX[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            invXtX[i][j] = mat[i][j+3];
        }
    }
    double t_crit = getTCritical(alpha, n - 3);
    for (size_t k = 1; k <= steps; ++k) {
        double t_future = static_cast<double>(n + k - 1);
        double t2_future = t_future * t_future;
        double point = a + b * t_future + c * t2_future;
        double x[3] = {1.0, t_future, t2_future};
        double quad_form = 0.0;
        for (int i = 0; i < 3; ++i) {
            double temp = 0.0;
            for (int j = 0; j < 3; ++j) {
                temp += x[j] * invXtX[j][i];
            }
            quad_form += temp * x[i];
        }
        double se_pred = std_err * std::sqrt(1.0 + quad_form);
        double half_width = t_crit * se_pred;
        double lower = point - half_width;
        double upper = point + half_width;
        results.push_back({point, lower, upper});
    }
    return results;
}

std::vector<SeriesClass::Forecast> SeriesClass::forecastPolynomial3(size_t steps, double alpha) const {
    std::vector<Forecast> results;
    if (data.size() < 4) return results;
    size_t n = data.size();
    auto coeffs = fitPolynomial3();
    if (coeffs.size() != 4) return results;
    double a = coeffs[0], b = coeffs[1], c = coeffs[2], d = coeffs[3];
    double SSE = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i);
        double t2 = t * t;
        double t3 = t2 * t;
        double y_hat = a + b * t + c * t2 + d * t3;
        SSE += (data[i] - y_hat) * (data[i] - y_hat);
    }
    double MSE = SSE / (n - 4);
    double std_err = std::sqrt(MSE);
    double sum_t0 = static_cast<double>(n);
    double sum_t = 0.0, sum_t2 = 0.0, sum_t3 = 0.0, sum_t4 = 0.0, sum_t5 = 0.0, sum_t6 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i);
        double t2 = t * t;
        double t3 = t2 * t;
        double t4 = t3 * t;
        double t5 = t4 * t;
        double t6 = t5 * t;
        sum_t += t;
        sum_t2 += t2;
        sum_t3 += t3;
        sum_t4 += t4;
        sum_t5 += t5;
        sum_t6 += t6;
    }
    double XtX[4][4] = {
        {sum_t0, sum_t, sum_t2, sum_t3},
        {sum_t, sum_t2, sum_t3, sum_t4},
        {sum_t2, sum_t3, sum_t4, sum_t5},
        {sum_t3, sum_t4, sum_t5, sum_t6}
    };
    double mat[4][8];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            mat[i][j] = XtX[i][j];
            mat[i][j+4] = (i == j) ? 1.0 : 0.0;
        }
    }
    for (int i = 0; i < 4; ++i) {
        int pivot_row = i;
        for (int k = i + 1; k < 4; ++k) {
            if (std::abs(mat[k][i]) > std::abs(mat[pivot_row][i])) {
                pivot_row = k;
            }
        }
        if (pivot_row != i) {
            for (int j = 0; j < 8; ++j) {
                std::swap(mat[i][j], mat[pivot_row][j]);
            }
        }
        if (std::abs(mat[i][i]) < 1e-12) {
            return results;
        }
        double pivot = mat[i][i];
        for (int j = 0; j < 8; ++j) {
            mat[i][j] /= pivot;
        }
        for (int k = 0; k < 4; ++k) {
            if (k != i) {
                double factor = mat[k][i];
                for (int j = 0; j < 8; ++j) {
                    mat[k][j] -= factor * mat[i][j];
                }
            }
        }
    }
    double invXtX[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            invXtX[i][j] = mat[i][j+4];
        }
    }
    double t_crit = getTCritical(alpha, n - 4);
    for (size_t k = 1; k <= steps; ++k) {
        double t_future = static_cast<double>(n + k - 1);
        double t2_future = t_future * t_future;
        double t3_future = t2_future * t_future;
        double point = a + b * t_future + c * t2_future + d * t3_future;
        double x[4] = {1.0, t_future, t2_future, t3_future};
        double quad_form = 0.0;
        for (int i = 0; i < 4; ++i) {
            double temp = 0.0;
            for (int j = 0; j < 4; ++j) {
                temp += x[j] * invXtX[j][i];
            }
            quad_form += temp * x[i];
        }
        double se_pred = std_err * std::sqrt(1.0 + quad_form);
        double half_width = t_crit * se_pred;
        double lower = point - half_width;
        double upper = point + half_width;
        results.push_back({point, lower, upper});
    }
    return results;
}

std::vector<SeriesClass::Forecast> SeriesClass::forecastExponential(size_t steps, double alpha, double confidence_level) const {
    std::vector<Forecast> results;
    if (data.empty()) return results;
    size_t n = data.size();
    std::vector<double> smoothed = exponentialSmoothing(alpha);
    if (smoothed.empty()) return results;
    double level = smoothed.back();
    std::vector<double> residuals;
    residuals.reserve(n - 1);
    for (size_t i = 1; i < n; ++i) {
        double one_step_forecast = smoothed[i - 1];
        residuals.push_back(data[i] - one_step_forecast);
    }
    if (residuals.empty()) return results;
    double sum_sq = 0.0;
    double mean_res = std::accumulate(residuals.begin(), residuals.end(), 0.0) / residuals.size();
    for (double res : residuals) {
        sum_sq += (res - mean_res) * (res - mean_res);
    }
    double variance = sum_sq / (residuals.size() - 1);
    double std_err = std::sqrt(variance);
    double t_crit = getTCritical(1.0 - confidence_level, residuals.size() - 1);
    for (size_t k = 1; k <= steps; ++k) {
        double point = level;
        double se_pred = std_err * std::sqrt(1.0 + (1.0 - alpha) * (1.0 - alpha));
        double half_width = t_crit * se_pred * std::sqrt(static_cast<double>(k));
        double lower = point - half_width;
        double upper = point + half_width;
        results.push_back({point, lower, upper});
    }
    return results;
}

SeriesClass::ForecastBacktestResult SeriesClass::backtestLinear(double train_fraction) const {
    ForecastBacktestResult result;
    if (data.size() < 10) return result;
    size_t n = data.size();
    size_t split = static_cast<size_t>(train_fraction * n);
    if (split < 2 || n - split < 2) return result;
    std::vector<double> train_values(data.begin(), data.begin() + split);
    std::vector<std::string> train_times(timestamps.begin(), timestamps.begin() + split);
    SeriesClass train_series(train_values, train_times, name + "_train");
    auto fixed_forecasts = train_series.forecastLinear(n - split);
    result.fixed_points.reserve(fixed_forecasts.size());
    for (const auto& f : fixed_forecasts) {
        result.fixed_points.push_back(f.point);
    }
    result.last_fixed_interval = fixed_forecasts.back();
    result.adaptive_points.reserve(n - split);
    Forecast last_adapt;
    for (size_t k = 0; k < n - split; ++k) {
        size_t curr_end = split + k;
        std::vector<double> curr_values(data.begin(), data.begin() + curr_end);
        std::vector<std::string> curr_times(timestamps.begin(), timestamps.begin() + curr_end);
        SeriesClass curr_series(curr_values, curr_times, name + "_curr");
        auto one_step = curr_series.forecastLinear(1);
        result.adaptive_points.push_back(one_step[0].point);
        if (k == n - split - 1) {
            last_adapt = one_step[0];
        }
    }
    result.last_adaptive_interval = last_adapt;
    double sum_sq_fixed = 0.0, sum_sq_adapt = 0.0;
    for (size_t i = 0; i < n - split; ++i) {
        double actual = data[split + i];
        sum_sq_fixed += (actual - result.fixed_points[i]) * (actual - result.fixed_points[i]);
        sum_sq_adapt += (actual - result.adaptive_points[i]) * (actual - result.adaptive_points[i]);
    }
    result.mse_fixed = sum_sq_fixed / (n - split);
    result.mse_adapt = sum_sq_adapt / (n - split);
    return result;
}

SeriesClass::ForecastBacktestResult SeriesClass::backtestPolynomial3(double train_fraction) const {
    ForecastBacktestResult result;
    if (data.size() < 10) return result;
    size_t n = data.size();
    size_t split = static_cast<size_t>(train_fraction * n);
    if (split < 4 || n - split < 2) return result;
    std::vector<double> train_values(data.begin(), data.begin() + split);
    std::vector<std::string> train_times(timestamps.begin(), timestamps.begin() + split);
    SeriesClass train_series(train_values, train_times, name + "_train");
    auto fixed_forecasts = train_series.forecastPolynomial3(n - split);
    result.fixed_points.reserve(fixed_forecasts.size());
    for (const auto& f : fixed_forecasts) {
        result.fixed_points.push_back(f.point);
    }
    result.last_fixed_interval = fixed_forecasts.back();
    result.adaptive_points.reserve(n - split);
    Forecast last_adapt;
    for (size_t k = 0; k < n - split; ++k) {
        size_t curr_end = split + k;
        std::vector<double> curr_values(data.begin(), data.begin() + curr_end);
        std::vector<std::string> curr_times(timestamps.begin(), timestamps.begin() + curr_end);
        SeriesClass curr_series(curr_values, curr_times, name + "_curr");
        auto one_step = curr_series.forecastPolynomial3(1);
        result.adaptive_points.push_back(one_step[0].point);
        if (k == n - split - 1) {
            last_adapt = one_step[0];
        }
    }
    result.last_adaptive_interval = last_adapt;
    double sum_sq_fixed = 0.0, sum_sq_adapt = 0.0;
    for (size_t i = 0; i < n - split; ++i) {
        double actual = data[split + i];
        sum_sq_fixed += (actual - result.fixed_points[i]) * (actual - result.fixed_points[i]);
        sum_sq_adapt += (actual - result.adaptive_points[i]) * (actual - result.adaptive_points[i]);
    }
    result.mse_fixed = sum_sq_fixed / (n - split);
    result.mse_adapt = sum_sq_adapt / (n - split);
    return result;
}

SeriesClass::ForecastBacktestResult SeriesClass::backtestExponential(double alpha, double train_fraction) const {
    ForecastBacktestResult result;
    if (data.size() < 10) return result;
    size_t n = data.size();
    size_t split = static_cast<size_t>(train_fraction * n);
    if (split < 2 || n - split < 2) return result;
    std::vector<double> train_values(data.begin(), data.begin() + split);
    std::vector<std::string> train_times(timestamps.begin(), timestamps.begin() + split);
    SeriesClass train_series(train_values, train_times, name + "_train");
    std::vector<double> train_smoothed = train_series.exponentialSmoothing(alpha);
    double fixed_level = train_smoothed.back();
    result.fixed_points.resize(n - split, fixed_level);
    auto fixed_forecasts = train_series.forecastExponential(n - split, alpha);
    result.last_fixed_interval = fixed_forecasts.back();
    result.adaptive_points.reserve(n - split);
    Forecast last_adapt;
    for (size_t k = 0; k < n - split; ++k) {
        size_t curr_end = split + k;
        std::vector<double> curr_values(data.begin(), data.begin() + curr_end);
        std::vector<std::string> curr_times(timestamps.begin(), timestamps.begin() + curr_end);
        SeriesClass curr_series(curr_values, curr_times, name + "_curr");
        auto one_step = curr_series.forecastExponential(1, alpha);
        result.adaptive_points.push_back(one_step[0].point);
        if (k == n - split - 1) {
            last_adapt = one_step[0];
        }
    }
    result.last_adaptive_interval = last_adapt;
    double sum_sq_fixed = 0.0, sum_sq_adapt = 0.0;
    for (size_t i = 0; i < n - split; ++i) {
        double actual = data[split + i];
        sum_sq_fixed += (actual - result.fixed_points[i]) * (actual - result.fixed_points[i]);
        sum_sq_adapt += (actual - result.adaptive_points[i]) * (actual - result.adaptive_points[i]);
    }
    result.mse_fixed = sum_sq_fixed / (n - split);
    result.mse_adapt = sum_sq_adapt / (n - split);
    return result;
}