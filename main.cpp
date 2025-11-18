#include <iostream>
#include <string>
#include <iomanip>
#include <limits>

#include "Records/RecordClass.h"
#include "Records/RecordControl.h"

#include "Statistics/StatisticsClass.h"
#include "Statistics/StatisticsControl.h"

#include "Regression/RegressionControl.h"
#include "Regression/RegressionClass.h"

#include "Series/SeriesClass.h"
#include "Series/SeriesControl.h"

const std::string RED_COLOR = "\033[31m";
const std::string GREEN_COLOR = "\033[32m";
const std::string YELLOW_COLOR = "\033[33m";
const std::string BLUE_COLOR = "\033[34m";
const std::string MAGENTA_COLOR = "\033[35m";
const std::string CYAN_COLOR = "\033[36m";
const std::string WHITE_COLOR = "\033[37m";
const std::string RESET = "\033[0m";
const std::string BOLD = "\033[1m";

static void displayMenu() {
    std::cout << "\n\n" << GREEN_COLOR << "=== --------------------------------------------------------------------------------------- ===" << RESET;
    std::cout << "\n" << BLUE_COLOR << "====== COMMODITY ANALYSIS MENU ======" << RESET << "\n";
    std::cout << "1. Show prices for specific date\n";
    std::cout << "2. Show dataset overview\n";
    std::cout << "3. Show full statistics report\n";
    std::cout << "4. Compare commodities\n";
    
    std::cout << "\n" << GREEN_COLOR << "===== CORRELATION ANALYSIS MENU =====" << RESET << "\n";
    std::cout << "5. Analyze correlations with Gold\n";
    std::cout << "6. Generate correlation graphs\n";
    std::cout << "7. Generate correlation matrix\n";
    
    std::cout << "\n" << MAGENTA_COLOR << "======== REGRESSION ANALYSIS ========" << RESET << "\n";
    std::cout << "8. Run Silver-Gold regression\n";
    std::cout << "9. Run Copper-Gold regression\n";
    std::cout << "10. Run custom commodity regression\n";
    std::cout << "11. Run multi-regression analysis (all predictors)\n";
    std::cout << "12. Run multi-regression analysis (LNG only)\n";
    std::cout << "13. Run multi-regression analysis (plants)\n";
    std::cout << "14. Run multi-regression analysis (metals)\n";
    std::cout << "15. Run multi-regression analysis (significant predictors++)\n";
    std::cout << "16. Run multi-regression analysis (significant predictors, two most)\n";
    
    std::cout << "\n" << CYAN_COLOR << "======== DISTRIBUTION PLOTS =========" << RESET << "\n";
    std::cout << "17. Generate histograms for all commodities\n";
    std::cout << "18. Generate boxplots for all commodities\n";
    
    std::cout << "\n" << YELLOW_COLOR << "========= FICTIVE PARAMETER =========" << RESET << "\n";
    std::cout << "19. Show fictive parameter details\n";
    std::cout << "20. Run multi-regression analysis (all predictors + fictive) ADDITIVE\n";
    std::cout << "21. Run multi-regression analysis (LNG only + fictive) ADDITIVE\n";
    std::cout << "22. Run multi-regression analysis (plants + fictive) ADDITIVE\n";
    std::cout << "23. Run multi-regression analysis (metals + fictive) ADDITIVE\n";
    std::cout << "24. Run multi-regression analysis (significant predictors++ + fictive) ADDITIVE\n";
    std::cout << "25. Run multi-regression analysis (significant predictors, two most + fictive) ADDITIVE\n";
    std::cout << "26. Run multi-regression analysis (all predictors + fictive) MULT\n";
    std::cout << "27. Run multi-regression analysis (LNG only + fictive) MULT\n";
    std::cout << "28. Run multi-regression analysis (plants + fictive) MULT\n";
    std::cout << "29. Run multi-regression analysis (metals + fictive) MULT\n";
    std::cout << "30. Run multi-regression analysis (significant predictors++ + fictive) MULT\n";
    std::cout << "31. Run multi-regression analysis (significant predictors, two most + fictive) MULT\n";
    
    std::cout << "\n" << WHITE_COLOR << "========= TIME SERIES ANALYSIS ========" << RESET << "\n";
    std::cout << "32. Show time series information\n";
    std::cout << "33. Detect and process anomalies (Irwin criterion)\n";
    std::cout << "34. Analyze trend presence\n";
    std::cout << "35. Compare all smoothing methods\n";
    std::cout << "36. Generate all time series plots\n";
    std::cout << "37. Time series decomposition\n";
    
    std::cout << "\n" << BOLD << CYAN_COLOR << "========= GROWTH CURVE ANALYSIS ========" << RESET << "\n";
    std::cout << "39. Fit and compare growth curves\n";
    
    std::cout << "\n" << BOLD << MAGENTA_COLOR << "====== RESIDUAL COMPONENT ANALYSIS ======" << RESET << "\n";
    std::cout << "40. Analyze residual component adequacy\n";

    std::cout << "\n" << BOLD << YELLOW_COLOR << "========= FORECAST ANALYSIS ========" << RESET << "\n";
    std::cout << "41. Run forecast analysis (3 and 7 steps ahead)\n";
    std::cout << "42. Run Brown's adaptive model forecast analysis\n";
    
    std::cout << "\n0. Exit program\n";
    std::cout << "Enter your choice (0-42): ";
}

static void displayItemsMenu(const std::vector<std::string>& items) {
    std::cout << "\n" << CYAN_COLOR << "=== AVAILABLE COMMODITIES ===" << RESET << "\n";
    for (size_t i = 0; i < items.size(); ++i) {
        std::cout << i+1 << ". " << items[i] << "\n";
    }
    std::cout << "0. Cancel\n";
    std::cout << "Select commodity: ";
}

static void displayTimeSeriesMenu(const std::vector<std::string>& items) {
    std::cout << "\n" << WHITE_COLOR << "=== AVAILABLE TIME SERIES ===" << RESET << "\n";
    for (size_t i = 0; i < items.size(); ++i) {
        std::string commodityLabel;
        if (items[i] == RecordClass::WTI_OIL) commodityLabel = "Crude Oil";
        else if (items[i] == RecordClass::GOLD) commodityLabel = "Gold";
        else if (items[i] == RecordClass::SILVER) commodityLabel = "Silver";
        else if (items[i] == RecordClass::NATURAL_GAS) commodityLabel = "Natural Gas";
        else if (items[i] == RecordClass::CORN) commodityLabel = "Corn";
        else if (items[i] == RecordClass::WHEAT) commodityLabel = "Wheat";
        else if (items[i] == RecordClass::SOYBEAN) commodityLabel = "Soybean";
        else if (items[i] == RecordClass::COPPER) commodityLabel = "Copper";
        else if (items[i] == RecordClass::PLATINUM) commodityLabel = "Platinum";
        else if (items[i] == RecordClass::PALLADIUM) commodityLabel = "Palladium";
        else commodityLabel = items[i];
        
        std::cout << i+1 << ". " << commodityLabel << "\n";
    }
    std::cout << "0. Cancel\n";
    std::cout << "Select time series: ";
}

static void runCustomRegression(const StatisticsControl& statsControl, const RegressionControl& regControl) {
    auto items = statsControl.getAvailableTypes();
    if (items.empty()) {
        std::cout << RED_COLOR << "No commodities available for analysis!\n" << RESET;
        return;
    }

    displayItemsMenu(items);
    
    int choice;
    if (!(std::cin >> choice)) {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << RED_COLOR << "Invalid input! Please enter a number.\n" << RESET;
        return;
    }

    if (choice == 0) return;
    if (choice < 0 || choice > static_cast<int>(items.size())) {
        std::cout << RED_COLOR << "Invalid selection!\n" << RESET;
        return;
    }

    const std::string& selectedCommodity = items[choice-1];
    std::string commodityLabel;
    
    if (selectedCommodity == RecordClass::WTI_OIL) commodityLabel = "Crude Oil";
    else if (selectedCommodity == RecordClass::SILVER) commodityLabel = "Silver";
    else if (selectedCommodity == RecordClass::NATURAL_GAS) commodityLabel = "Natural Gas";
    else if (selectedCommodity == RecordClass::CORN) commodityLabel = "Corn";
    else if (selectedCommodity == RecordClass::WHEAT) commodityLabel = "Wheat";
    else if (selectedCommodity == RecordClass::SOYBEAN) commodityLabel = "Soybean";
    else if (selectedCommodity == RecordClass::COPPER) commodityLabel = "Copper";
    else if (selectedCommodity == RecordClass::PLATINUM) commodityLabel = "Platinum";
    else if (selectedCommodity == RecordClass::PALLADIUM) commodityLabel = "Palladium";
    else commodityLabel = selectedCommodity;

    regControl.runCommodityRegression(selectedCommodity, commodityLabel);
}

static SeriesControl selectTimeSeries(const RecordClass& data) {
    std::vector<std::string> commodities = {
        RecordClass::WTI_OIL,
        RecordClass::GOLD,
        RecordClass::SILVER,
        RecordClass::NATURAL_GAS,
        RecordClass::CORN,
        RecordClass::WHEAT,
        RecordClass::SOYBEAN,
        RecordClass::COPPER,
        RecordClass::PLATINUM,
        RecordClass::PALLADIUM
    };
    
    std::vector<std::string> commodityLabels;
    for (const auto& commodity : commodities) {
        if (commodity == RecordClass::WTI_OIL) commodityLabels.push_back("Crude Oil");
        else if (commodity == RecordClass::GOLD) commodityLabels.push_back("Gold");
        else if (commodity == RecordClass::SILVER) commodityLabels.push_back("Silver");
        else if (commodity == RecordClass::NATURAL_GAS) commodityLabels.push_back("Natural Gas");
        else if (commodity == RecordClass::CORN) commodityLabels.push_back("Corn");
        else if (commodity == RecordClass::WHEAT) commodityLabels.push_back("Wheat");
        else if (commodity == RecordClass::SOYBEAN) commodityLabels.push_back("Soybean");
        else if (commodity == RecordClass::COPPER) commodityLabels.push_back("Copper");
        else if (commodity == RecordClass::PLATINUM) commodityLabels.push_back("Platinum");
        else if (commodity == RecordClass::PALLADIUM) commodityLabels.push_back("Palladium");
    }
    
    displayTimeSeriesMenu(commodityLabels);
    
    int choice;
    if (!(std::cin >> choice)) {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << RED_COLOR << "Invalid input! Please enter a number.\n" << RESET;
        return SeriesControl(data, RecordClass::GOLD);
    }

    if (choice == 0) return SeriesControl(data, RecordClass::GOLD);
    if (choice < 0 || choice > static_cast<int>(commodities.size())) {
        std::cout << RED_COLOR << "Invalid selection! Using Gold as default.\n" << RESET;
        return SeriesControl(data, RecordClass::GOLD);
    }

    const std::string& selectedCommodity = commodities[choice-1];
    std::cout << GREEN_COLOR << "Selected time series: " << commodityLabels[choice-1] << RESET << std::endl;
    
    return SeriesControl(data, selectedCommodity);
}

static void runTimeSeriesAnalysis(const RecordClass& data) {
    SeriesControl seriesControl = selectTimeSeries(data);
    
    int subChoice;
    std::cout << "\n" << WHITE_COLOR << "=== TIME SERIES ANALYSIS OPTIONS ===" << RESET << "\n";
    std::cout << "1. Show series information\n";
    std::cout << "2. Detect and process anomalies\n";
    std::cout << "3. Analyze trend presence\n";
    std::cout << "4. Apply all smoothing methods and generate plots\n";
    std::cout << "5. Time series decomposition\n";
    std::cout << "6. Analyze growth curves\n";
    std::cout << "0. Back to main menu\n";
    std::cout << "Enter your choice: ";
    
    if (!(std::cin >> subChoice)) {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << RED_COLOR << "Invalid input!\n" << RESET;
        return;
    }
    
    switch (subChoice) {
        case 1:
            seriesControl.displaySeriesInfo();
            break;
            
        case 2:
            seriesControl.processAnomalies(1.5);
            break;
            
        case 3:
            seriesControl.analyzeTrends();
            break;
            
        case 4: {
            seriesControl.processAnomalies(1.5);
            auto ma5 = seriesControl.movingAverage5();
            auto wma5 = seriesControl.weightedMovingAverage5();
            auto wma7 = seriesControl.weightedMovingAverage7();
            auto ca12 = seriesControl.chronologicalAverage12();
            auto expSmooth = seriesControl.exponentialSmoothing(0.3);
            
            std::vector<SeriesClass> smoothedSeries = {ma5, wma5, wma7, ca12, expSmooth};
            seriesControl.displayComparison(smoothedSeries);
            seriesControl.plotAllSeries(smoothedSeries);
            break;
        }
            
        case 5:
            seriesControl.plotDecomposition(365);
            break;
            
        case 6:
            seriesControl.analyzeGrowthCurves();
            break;
            
        case 0:
            return;
            
        default:
            std::cout << RED_COLOR << "Invalid option!\n" << RESET;
    }
}

static void runResidualAnalysis(const RecordClass& data) {
    std::cout << MAGENTA_COLOR << "\n=== RESIDUAL COMPONENT ANALYSIS ===" << RESET << std::endl;
    
    SeriesControl seriesControl = selectTimeSeries(data);
    
    std::cout << CYAN_COLOR << "Performing time series decomposition..." << RESET << std::endl;
    
    int period;
    std::cout << "Enter period for decomposition (e.g., 12 for monthly, 365 for yearly): ";
    if (!(std::cin >> period)) {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << RED_COLOR << "Invalid input! Using default period 365." << RESET << std::endl;
        period = 365;
    }
    
    seriesControl.plotDecomposition(period);
    
    std::cout << CYAN_COLOR << "\nAnalyzing residual component adequacy..." << RESET << std::endl;
    
    auto decomposition = seriesControl.getSeries().decomposeTimeSeries(period);
    
    SeriesClass residualsSeries(decomposition.finalResidual, 
                               seriesControl.getSeries().getTimestamps(), 
                               seriesControl.getSeries().getName() + " - Residuals");
    
    auto analysis = residualsSeries.analyzeResiduals();
    
    std::cout << "\n" << GREEN_COLOR << "=== RESIDUAL ANALYSIS RESULTS ===" << RESET << std::endl;
    std::cout << "Residual series: " << residualsSeries.getName() << std::endl;
    std::cout << "Number of points: " << residualsSeries.size() << std::endl;
    
    std::cout << "\n" << YELLOW_COLOR << "1. RANDOMNESS TESTS:" << RESET << std::endl;
    std::cout << "Turning points count: " << analysis.turningPointsCount << std::endl;
    std::cout << "Random by turning points: " << (analysis.isRandomByTurningPoints ? GREEN_COLOR + "YES" + RESET : RED_COLOR + "NO" + RESET) << std::endl;
    std::cout << "Series count: " << analysis.seriesCount << std::endl;
    std::cout << "Random by series test: " << (analysis.isRandomBySeries ? GREEN_COLOR + "YES" + RESET : RED_COLOR + "NO" + RESET) << std::endl;
    
    std::cout << "\n" << YELLOW_COLOR << "2. NORMALITY TESTS:" << RESET << std::endl;
    std::cout << "Skewness: " << analysis.skewness << " (should be near 0)" << std::endl;
    std::cout << "Kurtosis: " << analysis.kurtosis << " (should be near 0)" << std::endl;
    std::cout << "Normal by moments: " << (analysis.isNormalByMoments ? GREEN_COLOR + "YES" + RESET : RED_COLOR + "NO" + RESET) << std::endl;
    std::cout << "RS-statistic: " << analysis.rSStatistic << " (should be 7.5-8.5)" << std::endl;
    std::cout << "Normal by RS-test: " << (analysis.isNormalByRS ? GREEN_COLOR + "YES" + RESET : RED_COLOR + "NO" + RESET) << std::endl;
    
    std::cout << "\n" << YELLOW_COLOR << "3. ZERO MEAN TEST:" << RESET << std::endl;
    std::cout << "Mean: " << analysis.mean << std::endl;
    std::cout << "T-statistic: " << analysis.tStatistic << " (should be < 2.0)" << std::endl;
    std::cout << "Has zero mean: " << (analysis.hasZeroMean ? GREEN_COLOR + "YES" + RESET : RED_COLOR + "NO" + RESET) << std::endl;
    
    std::cout << "\n" << YELLOW_COLOR << "4. INDEPENDENCE TEST:" << RESET << std::endl;
    std::cout << "Durbin-Watson statistic: " << analysis.durbinWatsonStatistic 
              << " (should be 1.5-2.5)" << std::endl;
    std::cout << "Independent: " << (analysis.isIndependent ? GREEN_COLOR + "YES" + RESET : RED_COLOR + "NO" + RESET) << std::endl;
    
    std::cout << "\n" << BOLD << MAGENTA_COLOR << "=== FINAL CONCLUSION ===" << RESET << std::endl;
    std::cout << "Randomness (turning points): " << (analysis.isRandomByTurningPoints ? GREEN_COLOR + "PASS" + RESET : RED_COLOR + "FAIL" + RESET) << std::endl;
    std::cout << "Randomness (series test): " << (analysis.isRandomBySeries ? GREEN_COLOR + "PASS" + RESET : RED_COLOR + "FAIL" + RESET) << std::endl;
    std::cout << "Normality (skewness/kurtosis): " << (analysis.isNormalByMoments ? GREEN_COLOR + "PASS" + RESET : RED_COLOR + "FAIL" + RESET) << std::endl;
    std::cout << "Normality (RS-test): " << (analysis.isNormalByRS ? GREEN_COLOR + "PASS" + RESET : RED_COLOR + "FAIL" + RESET) << std::endl;
    std::cout << "Zero mean (t-test): " << (analysis.hasZeroMean ? GREEN_COLOR + "PASS" + RESET : RED_COLOR + "FAIL" + RESET) << std::endl;
    std::cout << "Independence (Durbin-Watson): " << (analysis.isIndependent ? GREEN_COLOR + "PASS" + RESET : RED_COLOR + "FAIL" + RESET) << std::endl;
    
    std::cout << "\n" << BOLD << (analysis.isAdequate ? GREEN_COLOR : RED_COLOR) 
              << "OVERALL MODEL ADEQUACY: " << (analysis.isAdequate ? "ADEQUATE" : "NOT ADEQUATE") 
              << RESET << std::endl;
    
    if (analysis.isAdequate) {
        std::cout << GREEN_COLOR << "The trend model is adequate - residuals behave like white noise" << RESET << std::endl;
    } else {
        std::cout << RED_COLOR << "The trend model is not adequate - residuals show systematic patterns" << RESET << std::endl;
        std::cout << YELLOW_COLOR << "Recommendation: Consider using a different model or including additional factors" << RESET << std::endl;
    }
    
    std::cout << CYAN_COLOR << "\nGenerating residual analysis plots..." << RESET << std::endl;
    
    SeriesControl residualControl(seriesControl);
    residualControl.plotResiduals(residualsSeries);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << RED_COLOR << "Error: Missing input file\n" << RESET;
        std::cerr << "Usage: " << argv[0] << " <data_file.csv>\n";
        return 1;
    }

    std::cout << GREEN_COLOR << "\nLoading market data from " << argv[1] << "..." << RESET;
    RecordClass data;
    if (!data.loadFromCSV(argv[1])) {
        std::cerr << RED_COLOR << "Failed to load data file!\n" << RESET;
        return 1;
    }

    RecordControl recordControl(data);
    StatisticsClass stats(data);
    StatisticsControl statsControl(stats);
    RegressionControl regControl(statsControl);

    const std::vector<std::string> significantPredictors_set1 = {"NG=F_closing_price"};
    const std::vector<std::string> significantPredictors_set2 = {"ZS=F_closing_price", "ZC=F_closing_price", "ZW=F_closing_price"};
    const std::vector<std::string> significantPredictors_set3 = {"SI=F_closing_price", "HG=F_closing_price", "PA=F_closing_price", "PL=F_closing_price"};
    const std::vector<std::string> significantPredictors_set4 = {"SI=F_closing_price", "NG=F_closing_price", "HG=F_closing_price", "ZS=F_closing_price"};
    const std::vector<std::string> significantPredictors_set5 = {"NG=F_closing_price", "HG=F_closing_price"};
    const std::vector<std::string> significantPredictors_set0 = {"CL=F_closing_price", "GC=F_closing_price", "SI=F_closing_price", "NG=F_closing_price", "ZC=F_closing_price", "ZW=F_closing_price", "ZS=F_closing_price", "HG=F_closing_price", "PL=F_closing_price", "PA=F_closing_price"};
    
    int choice = 0;
    std::string input;

    while (true) {
        displayMenu();
        if (!(std::cin >> choice)) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << RED_COLOR << "Invalid input! Please enter a number.\n" << RESET;
            continue;
        }
        std::cin.ignore();

        switch (choice) {
            case 1: { 
                auto dates = recordControl.getAvailableDatesFormatted();
                if (dates.empty()) {
                    std::cout << RED_COLOR << "No market data available!\n" << RESET;
                    break;
                }
                std::cout << "Enter trading date (e.g. " << dates.front() << "): ";
                std::getline(std::cin, input);
                recordControl.displayDataForDate(input);
                break;
            }

            case 2:
                recordControl.displayDataSummary();
                break;

            case 3:
                statsControl.showFullReport();
                break;

            case 4:
                statsControl.showComparativeAnalysis();
                break;

            case 5:
                statsControl.showGoldCorrelations();
                break;

            case 6:
                std::cout << CYAN_COLOR << "\nGenerating scatter plots..." << RESET;
                statsControl.generateScatterPlotsWithGNUplot();
                break;

            case 7:
                std::cout << CYAN_COLOR << "\nGenerating correlation matrix..." << RESET << "\n";
                statsControl.generateCorrelationMatrix();
                break;

            case 8:
                regControl.runSilverGoldRegression();
                break;

            case 9:
                regControl.runCopperGoldRegression();
                break;

            case 10:
                runCustomRegression(statsControl, regControl);
                break;

            case 11:
                regControl.runMultipleRegressionAllCommodities();
                break;

            case 12:
                regControl.runMultipleRegressionSelected(significantPredictors_set1);
                break;
            
            case 13:
                regControl.runMultipleRegressionSelected(significantPredictors_set2);
                break;

            case 14:
                regControl.runMultipleRegressionSelected(significantPredictors_set3);
                break;

            case 15:
                regControl.runMultipleRegressionSelected(significantPredictors_set4);
                break;

            case 16:
                regControl.runMultipleRegressionSelected(significantPredictors_set5);
                break;

            case 17:
                std::cout << CYAN_COLOR << "\nGenerating histograms..." << RESET << "\n";
                statsControl.generateHistograms();
                break;

            case 18:
                std::cout << CYAN_COLOR << "\nGenerating boxplots..." << RESET << "\n";
                statsControl.generateBoxplots();
                break;

            case 19:
                recordControl.displaySanctionsInfo();
                break;

            case 20:
                regControl.runMultipleRegressionWithDummy(significantPredictors_set0, false);
                break;

            case 21:
                regControl.runMultipleRegressionWithDummy(significantPredictors_set1, false);
                break;
            
            case 22:
                regControl.runMultipleRegressionWithDummy(significantPredictors_set2, false);
                break;
            
            case 23:
                regControl.runMultipleRegressionWithDummy(significantPredictors_set3, false);
                break;
            
            case 24:
                regControl.runMultipleRegressionWithDummy(significantPredictors_set4, false);
                break;
            
            case 25:
                regControl.runMultipleRegressionWithDummy(significantPredictors_set5, false);
                break;

            case 26:
                regControl.runMultipleRegressionWithDummy(significantPredictors_set0, true);
                break;

            case 27:
                regControl.runMultipleRegressionWithDummy(significantPredictors_set1, true);
                break;
            
            case 28:
                regControl.runMultipleRegressionWithDummy(significantPredictors_set2, true);
                break;
            
            case 29:
                regControl.runMultipleRegressionWithDummy(significantPredictors_set3, true);
                break;
            
            case 30:
                regControl.runMultipleRegressionWithDummy(significantPredictors_set4, true);
                break;
            
            case 31:
                regControl.runMultipleRegressionWithDummy(significantPredictors_set5, true);
                break;

            case 32: {
                SeriesControl seriesControl = selectTimeSeries(data);
                seriesControl.displaySeriesInfo();
                break;
            }

            case 33: {
                SeriesControl seriesControl = selectTimeSeries(data);
                seriesControl.processAnomalies(1.5);
                break;
            }

            case 34: {
                SeriesControl seriesControl = selectTimeSeries(data);
                seriesControl.analyzeTrends();
                break;
            }

            case 35: {
                SeriesControl seriesControl = selectTimeSeries(data);
                auto ma5 = seriesControl.movingAverage5();
                auto wma5 = seriesControl.weightedMovingAverage5();
                auto wma7 = seriesControl.weightedMovingAverage7();
                auto ca12 = seriesControl.chronologicalAverage12();
                auto expSmooth = seriesControl.exponentialSmoothing(0.3);
                
                std::vector<SeriesClass> smoothedSeries = {ma5, wma5, wma7, ca12, expSmooth};
                seriesControl.displayComparison(smoothedSeries);
                break;
            }

            case 36: {
                SeriesControl seriesControl = selectTimeSeries(data);
                seriesControl.processAnomalies(1.5);
                auto ma5 = seriesControl.movingAverage5();
                auto wma5 = seriesControl.weightedMovingAverage5();
                auto wma7 = seriesControl.weightedMovingAverage7();
                auto ca12 = seriesControl.chronologicalAverage12();
                auto expSmooth = seriesControl.exponentialSmoothing(0.3);
                
                std::vector<SeriesClass> smoothedSeries = {ma5, wma5, wma7, ca12, expSmooth};
                seriesControl.plotAllSeries(smoothedSeries);
                break;
            }

            case 37: {
                SeriesControl seriesControl = selectTimeSeries(data);
                seriesControl.plotDecomposition(365);
                break;
            }

            case 39: {
                SeriesControl seriesControl = selectTimeSeries(data);
                seriesControl.analyzeGrowthCurves();
                break;
            }

            case 40: {
                runResidualAnalysis(data);
                break;
            }

            case 41: {
                SeriesControl seriesControl = selectTimeSeries(data);
                seriesControl.forecastAnalysis(3, 7);
                break;
            }

            case 42: {
                SeriesControl seriesControl = selectTimeSeries(data);
                seriesControl.forecastAnalysisBrown(0.3, 0.8);
                break;
            }

            case 0:
                std::cout << GREEN_COLOR << "\nClosing market analysis...\n" << RESET;
                return 0;

            default:
                std::cout << RED_COLOR << "Invalid option! Please choose 0-42.\n" << RESET;
        }
    }
}