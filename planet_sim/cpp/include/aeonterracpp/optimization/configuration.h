#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <variant>
#include <optional>
#include <any>
#include <mutex>
#include <fstream>
#include <iostream>

namespace aeonterracpp {

/**
 * @brief Parameter value type enumeration
 */
enum class ParamType {
    Int,
    Float,
    Double,
    Bool,
    String,
    Enum,
    Path,
    Color,
    Vector2,
    Vector3,
    Range,
    Array,
    Object
};

/**
 * @brief Parameter metadata for validation and UI
 */
struct ParamMetadata {
    std::string name;
    std::string description;
    ParamType type;
    bool required = false;
    bool advanced = false;
    std::string category;
    std::string units;
    
    // Optional validation constraints
    double minValue = 0;
    double maxValue = 0;
    double stepSize = 0;
    double defaultValue = 0;
    std::vector<std::string> enumOptions;
    std::string fileExtension;
    std::string regexPattern;
    
    // UI presentation hints
    bool hidden = false;
    std::string uiWidget;
    int uiOrder = 0;
    std::string uiHint;
};

/**
 * @brief Configuration parameter value container
 */
class ParamValue {
public:
    // Constructors for different value types
    ParamValue() : type_(ParamType::String), valueStr("") {}
    explicit ParamValue(int value) : type_(ParamType::Int), valueInt(value) {}
    explicit ParamValue(float value) : type_(ParamType::Float), valueFloat(value) {}
    explicit ParamValue(double value) : type_(ParamType::Double), valueDouble(value) {}
    explicit ParamValue(bool value) : type_(ParamType::Bool), valueBool(value) {}
    explicit ParamValue(const std::string& value) : type_(ParamType::String), valueStr(value) {}
    explicit ParamValue(const char* value) : type_(ParamType::String), valueStr(std::string(value)) {}
    
    // Set value
    template<typename T>
    void setValue(const T& value);
    
    // Get value
    template<typename T>
    T getValue() const;
    
    // Get value type
    ParamType getType() const { return type_; }
    
    // Convert to string representation
    std::string toString() const;
    
    // Parse from string representation
    bool fromString(const std::string& str, ParamType type);
    
private:
    ParamType type_;
    
    // Use individual member variables instead of std::variant to avoid C++17 issues
    int valueInt = 0;
    float valueFloat = 0.0f;
    double valueDouble = 0.0;
    bool valueBool = false;
    std::string valueStr;
};

/**
 * @brief Configuration section containing related parameters
 */
class ConfigSection {
public:
    /**
     * @brief Construct a new Config Section
     * 
     * @param name Section name
     * @param description Section description
     */
    ConfigSection(const std::string& name, const std::string& description = "");
    
    /**
     * @brief Get section name
     * 
     * @return const std::string& Section name
     */
    const std::string& getName() const;
    
    /**
     * @brief Get section description
     * 
     * @return const std::string& Section description
     */
    const std::string& getDescription() const;
    
    /**
     * @brief Add parameter to section
     * 
     * @param key Parameter key
     * @param value Parameter value
     * @param metadata Parameter metadata
     * @return true Parameter added successfully
     * @return false Parameter already exists
     */
    bool addParam(const std::string& key, const ParamValue& value, 
                  const ParamMetadata& metadata = ParamMetadata());
    
    /**
     * @brief Check if parameter exists
     * 
     * @param key Parameter key
     * @return true Parameter exists
     * @return false Parameter does not exist
     */
    bool hasParam(const std::string& key) const;
    
    /**
     * @brief Get parameter value
     * 
     * @param key Parameter key
     * @return const ParamValue& Parameter value
     * @throws std::out_of_range if parameter does not exist
     */
    const ParamValue& getParam(const std::string& key) const;
    
    /**
     * @brief Get parameter value
     * 
     * @param key Parameter key
     * @return ParamValue& Parameter value reference
     * @throws std::out_of_range if parameter does not exist
     */
    ParamValue& getParam(const std::string& key);
    
    /**
     * @brief Get parameter metadata
     * 
     * @param key Parameter key
     * @return const ParamMetadata& Parameter metadata
     * @throws std::out_of_range if parameter does not exist
     */
    const ParamMetadata& getMetadata(const std::string& key) const;
    
    /**
     * @brief Set parameter value
     * 
     * @param key Parameter key
     * @param value Parameter value
     * @return true Value set successfully
     * @return false Parameter does not exist
     */
    bool setParam(const std::string& key, const ParamValue& value);
    
    /**
     * @brief Get all parameter keys
     * 
     * @return std::vector<std::string> Parameter keys
     */
    std::vector<std::string> getParamKeys() const;
    
    /**
     * @brief Remove parameter
     * 
     * @param key Parameter key
     * @return true Parameter removed successfully
     * @return false Parameter does not exist
     */
    bool removeParam(const std::string& key);
    
private:
    std::string name_;
    std::string description_;
    std::unordered_map<std::string, ParamValue> params_;
    std::unordered_map<std::string, ParamMetadata> metadata_;
};

/**
 * @brief Configuration manager for simulation parameters
 * 
 * This class manages configuration parameters for all simulation
 * components, including loading/saving configurations and validation.
 */
class ConfigManager {
public:
    /**
     * @brief Get singleton instance
     * 
     * @return ConfigManager& Singleton instance
     */
    static ConfigManager& getInstance();
    
    /**
     * @brief Create a new section
     * 
     * @param name Section name
     * @param description Section description
     * @return ConfigSection& Reference to created section
     */
    ConfigSection& createSection(const std::string& name, const std::string& description = "");
    
    /**
     * @brief Get section
     * 
     * @param name Section name
     * @return ConfigSection& Reference to section
     * @throws std::out_of_range if section does not exist
     */
    ConfigSection& getSection(const std::string& name);
    
    /**
     * @brief Get section (const)
     * 
     * @param name Section name
     * @return const ConfigSection& Reference to section
     * @throws std::out_of_range if section does not exist
     */
    const ConfigSection& getSection(const std::string& name) const;
    
    /**
     * @brief Check if section exists
     * 
     * @param name Section name
     * @return true Section exists
     * @return false Section does not exist
     */
    bool hasSection(const std::string& name) const;
    
    /**
     * @brief Get all section names
     * 
     * @return std::vector<std::string> Section names
     */
    std::vector<std::string> getSectionNames() const;
    
    /**
     * @brief Remove section
     * 
     * @param name Section name
     * @return true Section removed successfully
     * @return false Section does not exist
     */
    bool removeSection(const std::string& name);
    
    /**
     * @brief Load configuration from file
     * 
     * @param filename Configuration file path
     * @return true Configuration loaded successfully
     * @return false Failed to load configuration
     */
    bool loadFromFile(const std::string& filename);
    
    /**
     * @brief Save configuration to file
     * 
     * @param filename Configuration file path
     * @return true Configuration saved successfully
     * @return false Failed to save configuration
     */
    bool saveToFile(const std::string& filename) const;
    
    /**
     * @brief Get parameter value
     * 
     * @tparam T Parameter value type
     * @param section Section name
     * @param key Parameter key
     * @param defaultValue Default value if parameter does not exist
     * @return T Parameter value
     */
    template<typename T>
    T getValue(const std::string& section, const std::string& key, const T& defaultValue) const;
    
    /**
     * @brief Set parameter value
     * 
     * @tparam T Parameter value type
     * @param section Section name
     * @param key Parameter key
     * @param value Parameter value
     * @return true Value set successfully
     * @return false Section or parameter does not exist
     */
    template<typename T>
    bool setValue(const std::string& section, const std::string& key, const T& value);
    
    /**
     * @brief Register change callback
     * 
     * @param section Section name
     * @param key Parameter key
     * @param callback Callback function
     * @return int Callback ID
     */
    int registerChangeCallback(const std::string& section, const std::string& key, 
                              std::function<void(const ParamValue&)> callback);
    
    /**
     * @brief Unregister change callback
     * 
     * @param callbackId Callback ID
     * @return true Callback unregistered successfully
     * @return false Callback ID not found
     */
    bool unregisterChangeCallback(int callbackId);
    
    /**
     * @brief Validate configuration
     * 
     * @return std::vector<std::string> Validation errors (empty if valid)
     */
    std::vector<std::string> validate() const;
    
    /**
     * @brief Clear all sections and parameters
     */
    void clear();
    
    /**
     * @brief Register a processor
     * 
     * This function registers system components that have configurable parameters
     * 
     * @param name Processor name
     * @param registerFunc Function to register parameters
     */
    void registerProcessor(const std::string& name, 
                         std::function<void(ConfigManager&)> registerFunc);
    
    /**
     * @brief Initialize all registered processors
     */
    void initializeProcessors();
    
private:
    // Private constructor for singleton
    ConfigManager();
    // Delete copy/move constructors and assignment operators
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;
    ConfigManager(ConfigManager&&) = delete;
    ConfigManager& operator=(ConfigManager&&) = delete;
    
    // Sections
    std::unordered_map<std::string, ConfigSection> sections_;
    
    // Change callbacks
    struct Callback {
        std::string section;
        std::string key;
        std::function<void(const ParamValue&)> func;
    };
    std::unordered_map<int, Callback> callbacks_;
    int nextCallbackId_ = 0;
    
    // Registered processors
    std::unordered_map<std::string, std::function<void(ConfigManager&)>> processors_;
    
    // Mutex for thread safety
    mutable std::mutex mutex_;
    
    // Notify callbacks about parameter change
    void notifyCallbacks(const std::string& section, const std::string& key, 
                        const ParamValue& value);
    
    // Parse value from string
    ParamValue parseValue(const std::string& value, ParamType type);
};

/**
 * @brief Configuration initializer for components
 * 
 * This class provides a standard way for system components
 * to register their configuration parameters.
 */
class ConfigInitializer {
public:
    /**
     * @brief Construct a new Config Initializer
     * 
     * @param configManager Configuration manager
     * @param sectionName Section name
     * @param sectionDescription Section description
     */
    ConfigInitializer(ConfigManager& configManager, 
                    const std::string& sectionName,
                    const std::string& sectionDescription = "");
    
    /**
     * @brief Get configuration section
     * 
     * @return ConfigSection& Reference to section
     */
    ConfigSection& getSection();
    
    /**
     * @brief Register integer parameter
     * 
     * @param key Parameter key
     * @param defaultValue Default value
     * @param description Parameter description
     * @return ConfigInitializer& Reference to this initializer
     */
    ConfigInitializer& registerInt(const std::string& key, int defaultValue,
                                 const std::string& description);
    
    /**
     * @brief Register float parameter
     * 
     * @param key Parameter key
     * @param defaultValue Default value
     * @param description Parameter description
     * @return ConfigInitializer& Reference to this initializer
     */
    ConfigInitializer& registerFloat(const std::string& key, float defaultValue,
                                   const std::string& description);
    
    /**
     * @brief Register boolean parameter
     * 
     * @param key Parameter key
     * @param defaultValue Default value
     * @param description Parameter description
     * @return ConfigInitializer& Reference to this initializer
     */
    ConfigInitializer& registerBool(const std::string& key, bool defaultValue,
                                  const std::string& description);
    
    /**
     * @brief Register string parameter
     * 
     * @param key Parameter key
     * @param defaultValue Default value
     * @param description Parameter description
     * @return ConfigInitializer& Reference to this initializer
     */
    ConfigInitializer& registerString(const std::string& key, const std::string& defaultValue,
                                    const std::string& description);
    
    /**
     * @brief Register enum parameter
     * 
     * @param key Parameter key
     * @param options Enum options
     * @param defaultValue Default value
     * @param description Parameter description
     * @return ConfigInitializer& Reference to this initializer
     */
    ConfigInitializer& registerEnum(const std::string& key, 
                                  const std::vector<std::string>& options,
                                  const std::string& defaultValue,
                                  const std::string& description);
    
    /**
     * @brief Register file path parameter
     * 
     * @param key Parameter key
     * @param defaultValue Default value
     * @param description Parameter description
     * @param fileExtension File extension filter
     * @return ConfigInitializer& Reference to this initializer
     */
    ConfigInitializer& registerPath(const std::string& key, const std::string& defaultValue,
                                  const std::string& description, 
                                  const std::string& fileExtension = "");
    
private:
    ConfigManager& configManager_;
    ConfigSection& section_;
};

/**
 * @brief User interface configuration
 * 
 * Configuration parameters related to the user interface
 * and parameter input controls.
 */
struct UIConfig {
    bool showAdvancedParams = false;      // Show advanced parameters
    bool compactDisplay = false;          // Use compact parameter display
    bool useCategoriesInUI = true;        // Group parameters by category
    bool showUnitsInLabels = true;        // Show units in parameter labels
    bool showHelpIcons = true;            // Show help icons for parameters
    bool enableSearchFilter = true;       // Enable parameter search/filtering
    bool showValidationImmediately = true; // Show validation errors immediately
    std::string theme = "default";        // UI theme
    bool useNativeFileDialogs = true;     // Use native file dialogs
    bool useGridLayout = true;            // Use grid layout for parameters
    int labelWidth = 200;                 // Width of parameter labels
    bool groupDefaultConfigurations = true; // Group default configurations
    bool showConfigHistory = true;        // Show configuration history
    bool allowParamPresets = true;        // Allow user-defined parameter presets
    
    /**
     * @brief Register configuration parameters
     * 
     * @param config Configuration manager
     */
    static void registerConfig(ConfigManager& config) {
        ConfigInitializer initializer(config, "ui", "User Interface Settings");
        
        initializer
            .registerBool("showAdvanced", false, "Show advanced parameters")
            .registerBool("compactDisplay", false, "Use compact parameter display")
            .registerBool("useCategories", true, "Group parameters by category")
            .registerBool("showUnits", true, "Show units in parameter labels")
            .registerBool("showHelp", true, "Show help icons for parameters")
            .registerBool("enableSearch", true, "Enable parameter search/filtering")
            .registerBool("immediateValidation", true, "Show validation errors immediately")
            .registerEnum("theme", {"default", "dark", "light", "high_contrast"}, 
                         "default", "UI theme")
            .registerBool("nativeFileDialogs", true, "Use native file dialogs")
            .registerBool("gridLayout", true, "Use grid layout for parameters")
            .registerInt("labelWidth", 200, "Width of parameter labels")
            .registerBool("groupDefaults", true, "Group default configurations")
            .registerBool("showHistory", true, "Show configuration history")
            .registerBool("allowPresets", true, "Allow user-defined parameter presets");
    }
};

/**
 * @brief Performance configuration
 * 
 * Configuration parameters related to performance optimization
 * and resource usage.
 */
struct PerformanceConfig {
    bool enableMultithreading = true;     // Enable multithreaded processing
    int numThreads = 0;                   // Number of threads (0 = auto)
    bool enableVectorization = true;      // Enable SIMD vectorization
    bool useGPUAcceleration = false;      // Use GPU acceleration if available
    bool enableCaching = true;            // Enable data caching
    int cacheSizeMB = 512;                // Cache size in MB
    bool enableProgressiveRendering = true; // Enable progressive rendering
    bool useLowResolutionPreview = true;  // Use low resolution for preview
    int maxVRAMUsageMB = 2048;            // Maximum VRAM usage in MB
    int targetFPS = 30;                   // Target frames per second
    bool enableAsyncLoading = true;       // Enable asynchronous loading
    bool useMemoryMapping = false;        // Use memory mapping for large datasets
    int maxRAMPercentage = 70;            // Maximum RAM usage as percentage
    std::string optimizationLevel = "balanced"; // Optimization level (performance vs quality)
    
    /**
     * @brief Register configuration parameters
     * 
     * @param config Configuration manager
     */
    static void registerConfig(ConfigManager& config) {
        ConfigInitializer initializer(config, "performance", "Performance Settings");
        
        initializer
            .registerBool("enableMultithreading", true, "Enable multithreaded processing")
            .registerInt("numThreads", 0, "Number of threads (0 = automatic)")
            .registerBool("enableVectorization", true, "Enable SIMD vectorization")
            .registerBool("useGPUAcceleration", false, "Use GPU acceleration if available")
            .registerBool("enableCaching", true, "Enable data caching")
            .registerInt("cacheSizeMB", 512, "Cache size")
            .registerBool("enableProgressiveRendering", true, "Enable progressive rendering")
            .registerBool("useLowResolutionPreview", true, "Use low resolution for preview")
            .registerInt("maxVRAMUsageMB", 2048, "Maximum VRAM usage")
            .registerInt("targetFPS", 30, "Target frames per second")
            .registerBool("enableAsyncLoading", true, "Enable asynchronous loading")
            .registerBool("useMemoryMapping", false, "Use memory mapping for large datasets")
            .registerInt("maxRAMPercentage", 70, "Maximum RAM usage")
            .registerEnum("optimizationLevel", 
                         {"quality", "balanced", "performance", "ultra"}, 
                         "balanced", "Optimization level");
    }
};

/**
 * @brief System configuration
 * 
 * Configuration parameters related to system integration
 * and technical settings.
 */
struct SystemConfig {
    std::string dataDirectory = "";       // Data directory path
    std::string tempDirectory = "";       // Temporary directory path
    std::string exportDirectory = "";     // Export directory path
    std::string logDirectory = "";        // Log directory path
    int logLevel = 2;                     // Log level (0=off, 1=error, 2=warning, 3=info, 4=debug)
    bool enableCrashReporting = true;     // Enable crash reporting
    bool autoSaveConfigs = true;          // Auto-save configurations
    int autoSaveIntervalMinutes = 10;     // Auto-save interval in minutes
    bool checkForUpdates = true;          // Check for updates on startup
    std::string lastOpenedFile = "";      // Last opened file path
    bool enableTelemetry = false;         // Enable anonymous usage data
    bool useSystemLocale = true;          // Use system locale
    std::string customLocale = "en_US";   // Custom locale
    std::string externalEditor = "";      // External editor path
    
    /**
     * @brief Register configuration parameters
     * 
     * @param config Configuration manager
     */
    static void registerConfig(ConfigManager& config) {
        ConfigInitializer initializer(config, "system", "System Settings");
        
        initializer
            .registerPath("dataDirectory", "", "Data directory path")
            .registerPath("tempDirectory", "", "Temporary directory path")
            .registerPath("exportDirectory", "", "Export directory path")
            .registerPath("logDirectory", "", "Log directory path")
            .registerInt("logLevel", 2, "Log level")
            .registerBool("enableCrashReporting", true, "Enable crash reporting")
            .registerBool("autoSaveConfigs", true, "Auto-save configurations")
            .registerInt("autoSaveInterval", 10, "Auto-save interval")
            .registerBool("checkForUpdates", true, "Check for updates on startup")
            .registerPath("lastOpenedFile", "", "Last opened file path")
            .registerBool("enableTelemetry", false, "Enable anonymous usage data")
            .registerBool("useSystemLocale", true, "Use system locale")
            .registerString("customLocale", "en_US", "Custom locale")
            .registerPath("externalEditor", "", "External editor path");
    }
};

} // namespace aeonterracpp