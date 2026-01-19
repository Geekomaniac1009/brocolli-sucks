#include <bits/stdc++.h>
#include <chrono>
using namespace std;

/* ---------------------- Constants and Globals ---------------------- */
constexpr int NUM_SLOTS = 720;
constexpr double P_MAX = 400.0;
constexpr double U_MAX = 20.0;
constexpr double P_S = 1.0; 
constexpr double EPSILON = 1e-5;
constexpr double SLOT_HOURS = 24.0 / NUM_SLOTS;
constexpr long double WATT_TO_KWH = SLOT_HOURS / 1000.0;
constexpr double TASK_PROFIT_FACTOR = 0.001*90; // dollar to INR conversion
constexpr int    SDC_WINDOW = 4;  
static double BATTERY_CHARGE_EFF  = 0.999;   
static double BATTERY_DISCHARGE_EFF  = 0.999; 

// --- New Tiered Cost & SDC Constants ---
constexpr double TIER1_LIMIT_W = 120.0*1000/NUM_SLOTS;   // Tier 1 Power Limit
constexpr double TIER2_LIMIT_W = 240.0*1000/NUM_SLOTS;  // Tier 2 Power Limit
long double SLA_LOSS_TOLERANCE = 10000000.0; // Try executing all tasks for now

// Virtual Battery Partitioning
constexpr double SOLAR_BATT_RATIO = 0.5; // 50% of total cap for Solar
constexpr double GRID_BATT_RATIO = 0.5;  // 50% of total cap for Grid Shaving

/* ---------------------- Benchmarking Modes ---------------------- */

enum GridStrategy {
    GRID_NONE,              // For Pure Solar (Grid tasks dropped/ignored)
    GRID_OPTIMAL_OFFLINE,   // Current approach: Time-shift based on offline plan (Cost Minimization)
    GRID_GREEDY,            // Run remaining tasks immediately in current slot (No Shifting)
    GRID_SDC_SHAVING        // Run remaining tasks using Battery SDC (Peak Shaving)
};

enum BatteryMode { NO_BATTERY, FINITE_BATTERY, INFINITE_BATTERY };
enum HeuristicMetric { HPF, HPDF, LPF }; 

enum GridHeuristic {
    APP_A_SOLAR_ONLY_FINITE,       // Solar Only + Finite Battery Levelling
    APP_B_GRID_ONLY_FINITE,        // Grid Only + SDC Levelling (No Solar)
    APP_C_HYBRID_GREEDY,           // Solar Offline + Greedy Grid (No shifting)
    APP_D_HYBRID_OPTIMAL,          // Solar Offline + Optimal Grid Offline (Infinite Batt assumption)
    APP_E_HYBRID_SDC_LIMITED,      // Solar Offline + SDC Limited Horizon Window
    APP_F_HYBRID_SDC_OPTIMAL       // Solar Offline + SDC Optimal (No window, infinite battery)
};

struct ExperimentConfig {
    string name;
    GridHeuristic solarHeuristic; // PURE_SOLAR_HPF, SOLAR_FIRST_LPF
    GridStrategy gridStrategy;
    double solarBatteryRatio; 
    double sdcBatteryRatio;
};

struct Task {
    int id, arrival, deadline;
    double util, profit;
    int is_sched = 0;
};

struct SlotState {
    // Resources
    double solarPower = 0.0; 
    // Grid Cost Tiers: [Base, Premium, Ultra]
    vector<double> gridCostTiers; 
    double solarUtilLimit = 0.0;
    
    // Offline Plan
    set<int> offlineSolarScheduled;
    set<int> offlineGridScheduled;

    // --- Runtime State ---
    double solarUsedUtil = 0.0;
    double gridUsedUtil = 0.0;
    
    // Debug/Logging for SDC
    double sdcGridBatteryLevel = 0.0;
};

/**
 * @brief Holds all final metrics for a single simulation run.
 */
struct SimResult {        
    int tasksOnSolar;
    int tasksOnGrid;
    int tasksDropped;
    double netRevenue;
    double taskEfficiency;    
    vector <double> slotWiseGridUsage;
    vector <double> slotWiseSolarUsage;
    vector <double> slotWiseSolarBatteryLevel;
};

vector<Task> tasks; 
vector<SlotState> slots(NUM_SLOTS + 144);
double batteryCapacity = 10000.0;
vector<double> offlineGridUsage(NUM_SLOTS + 144, 0.0); // store offline planned grid power per slot

// --- Simulation State Globals ---
double totalProfit = 0.0;
double totalGridCost = 0.0;
vector<double> batteryLevel(NUM_SLOTS + 144, 0.0); // For offline solar planning

using TaskQueue = priority_queue<pair<double, int>>;

/* ---------------------- Utility Functions ---------------------- */

inline double util_to_pow(double u) { 
    return P_S + (P_MAX - P_S) * pow((max(0.0, u) / U_MAX), 3);
}

inline double pow_to_util(double p) {
    if (p < P_S) {
        return 0.0;
    }
    double val = (p - P_S) / (P_MAX - P_S);
    if (val <= 0) {
        return 0.0;
    }
    return U_MAX * cbrt(val);
}

inline bool crosses_tier(double pPrev, double pNext) {
    return (pPrev <= TIER1_LIMIT_W && pNext > TIER1_LIMIT_W) ||
           (pPrev <= TIER2_LIMIT_W && pNext > TIER2_LIMIT_W);
}

// --- Tiered Cost Calculator ---
// Correctly calculates cost considering tier transitions
// Tiers: [0-120W]=Base, [120-240W]=Premium, [240+W]=Ultra
double get_tiered_cost(double oldPow, double newPow, const vector<double>& tiers) {
    // oldPow, newPow in W, tiers costing in INR/kWhr
    if (newPow <= oldPow) {
        return 0.0;
    }

    double cost = 0.0;
    if(newPow <= TIER1_LIMIT_W){
        cost += (newPow - oldPow) * tiers[0];
    }
    else if(oldPow < TIER1_LIMIT_W){
        if(newPow <= TIER2_LIMIT_W){
            cost += (TIER1_LIMIT_W - oldPow) * tiers[0];
            cost += (newPow - TIER1_LIMIT_W) * tiers[1];
        }
        else{
            cost += (TIER1_LIMIT_W - oldPow) * tiers[0];
            cost += (TIER2_LIMIT_W - TIER1_LIMIT_W) * tiers[1];
            cost += (newPow - TIER2_LIMIT_W) * tiers[2];
        }
    }
    else if(oldPow < TIER2_LIMIT_W){
        if(newPow <= TIER2_LIMIT_W){
            cost += (newPow - oldPow) * tiers[1];
        }
        else{
            cost += (TIER2_LIMIT_W - oldPow) * tiers[1];
            cost += (newPow - TIER2_LIMIT_W) * tiers[2];
        }
    }
    else{
        cost += (newPow - oldPow) * tiers[2];
    }

    return cost * WATT_TO_KWH; 
}

/* ---------------------- CSV File Parser ---------------------- */

class FileReader {
public:
    static bool readPower(const string& filename) {
        ifstream file(filename); 
        if (!file.is_open()) {
            cerr << "[Error] Could not open Power file: " << filename << endl;
            return false;
        }

        string line; 
        getline(file, line); // Skip header
        int t = 0;
        while (getline(file, line) && t < NUM_SLOTS) {
            stringstream ss(line); 
            string segment;
            getline(ss, segment, ','); // Skip timestamp
            getline(ss, segment, ','); // Power value
            try {
                slots[t].solarPower = stod(segment);
                slots[t].solarUtilLimit = pow_to_util(slots[t].solarPower);
            } catch (...) { 
                cerr << "[Error] Parsing error in Power CSV at slot " << t << endl;
                return false;
            }
            t++;
        }
        cout << "[Success] Loaded " << t << " solar power slots." << endl;
        return true;
    }

    static bool readTasks(const string& filename) {
        tasks.clear();
        ifstream file(filename); 
        if (!file.is_open()) {
            cerr << "[Error] Could not open Tasks file: " << filename << endl;
            return false;
        }

        string line; getline(file, line); // Skip header
        int current_id = 0;
        while (getline(file, line)) {
             stringstream ss(line); string segment;
             vector<string> parts;
             while (getline(ss, segment, ',')) {
                parts.push_back(segment);
             }
             if (parts.size() < 5) continue;
             try {
                Task t;
                t.id = current_id++;
                t.arrival = static_cast<int>(stod(parts[1])) - 1;
                t.deadline = static_cast<int>(stod(parts[2])) - 1; 
                t.util = stod(parts[3]);
                t.profit = stod(parts[4]) * TASK_PROFIT_FACTOR;

                if (t.arrival < NUM_SLOTS && t.deadline >= t.arrival && t.deadline < NUM_SLOTS && t.id < 20000) {
                    tasks.push_back(t);
                }
             } catch (...) { 
                cerr << "[Error] Parsing error in Tasks CSV at line " << current_id << endl;
                return false;
             }
        }
        cout << "[Success] Loaded " << tasks.size() << " offline tasks." << endl;
        return true;
    }

    static bool readGridCost(const string& filename) {
        ifstream file(filename); 
        if (!file.is_open()) {
            cerr << "[Error] Could not open Grid Cost file: " << filename << endl;
            return false;
        }

        string line; getline(file, line); 
        int t = 0;
        while (getline(file, line) && t < NUM_SLOTS) {
            stringstream ss(line); 
            string segment;
            getline(ss, segment, ','); 
            
            slots[t].gridCostTiers.clear();
            while(getline(ss, segment, ',')) {
                try {
                    slots[t].gridCostTiers.push_back(stod(segment));
                } catch(...) {}
            }
            while (slots[t].gridCostTiers.size() < 3) {
                if(slots[t].gridCostTiers.empty()) {
                    slots[t].gridCostTiers.push_back(5.0);
                }
                else {
                    slots[t].gridCostTiers.push_back(1.4 * slots[t].gridCostTiers.back());
                }
            }
            t++;
        }
        int lastLoaded = t - 1; 
        for (int k = lastLoaded + 1; k < NUM_SLOTS; ++k) {
            // Carry forward the last known cost tiers to the end of the simulation
            slots[k].gridCostTiers = slots[lastLoaded].gridCostTiers;
        }
        cout << "[Success] Loaded grid cost tiers for " << t << " slots." << endl;
        return true;
    }

    static vector<double> readActualPower(const string& filename) {
        vector<double> actualPower(NUM_SLOTS + 144, 0.0);

        ifstream file(filename); 
        string line; 
        getline(file, line);
        int t = 0;
        while (getline(file, line) && t < NUM_SLOTS) {
            stringstream ss(line); 
            string segment;
            getline(ss, segment, ',');
            getline(ss, segment, ',');
            try {
                actualPower[t] = stod(segment);
            } catch (...) {
                throw runtime_error("Error in parsing actual power CSV.");
            }
            t++;
        }
        cout << "[Success] Loaded actual power for " << t << " slots." << endl;
        return actualPower;
    }

    static bool readActualTasks(const string& filename, 
                                vector<Task>& out_tasks_actual, 
                                vector<vector<int>>& out_arrivalsActualBySlot) 
    {
        out_tasks_actual.clear();
        out_arrivalsActualBySlot.assign(NUM_SLOTS + 144, vector<int>());

        ifstream file(filename); 
        if (!file.is_open()) {
            cerr << "[Error] Could not open Actual Tasks file: " << filename << endl;
            return false;
        }

        string line; getline(file, line); 
        int current_id = 0;
        while (getline(file, line)) {
             stringstream ss(line); 
             string segment;
             vector<string> parts;
             while (getline(ss, segment, ',')) parts.push_back(segment);
             
             if (parts.size() < 5) continue;
             try {
                Task t;
                t.id = current_id++; 
                t.arrival = static_cast<int>(stod(parts[1])) - 1;
                t.deadline = static_cast<int>(stod(parts[2])) - 1; 
                t.util = stod(parts[3]);
                t.profit = stod(parts[4]) * TASK_PROFIT_FACTOR;
                
                if (t.arrival < NUM_SLOTS && t.deadline >= t.arrival && t.deadline < NUM_SLOTS && t.id < 25000) {
                    out_tasks_actual.push_back(t);
                    if (t.arrival >= 0) {
                        out_arrivalsActualBySlot[t.arrival].push_back(t.id);
                    }
                }
             } catch (...) { 
                cerr << "[Error] Parsing error in Actual Tasks CSV." << endl;
                return false;
             }
        }
        cout << "[Success] Loaded " << out_tasks_actual.size() << " actual tasks." << endl;
        return true;
    }
};

/* ---------------------- Scheduling Primitives ---------------------- */

void sched_task_solar(int tid, int slot) {
    if(tasks[tid].is_sched) {
        return;
    }
    slots[slot].offlineSolarScheduled.insert(tid);
    slots[slot].solarUsedUtil += tasks[tid].util;
    totalProfit += tasks[tid].profit;
    tasks[tid].is_sched = 1;
}

void sched_task_grid(int tid, int slot) {
    if(tasks[tid].is_sched) {
        return;
    }
    slots[slot].offlineGridScheduled.insert(tid);
    slots[slot].gridUsedUtil += tasks[tid].util;
    totalProfit += tasks[tid].profit;
    tasks[tid].is_sched = 1;
}

bool unsched_min_util_task_grid(int slot, TaskQueue& taskQueue, double& savedPower) {
    if (slots[slot].offlineGridScheduled.empty()) {
        savedPower = 0.0;
        return false;
    }

    int tid_to_drop = -1;
    double min_util = numeric_limits<double>::max();
    for (int tid : slots[slot].offlineGridScheduled) {
        if (tasks[tid].util < min_util) {
            min_util = tasks[tid].util;
            tid_to_drop = tid;
        }
    }
    if (tid_to_drop == -1) {
        savedPower = 0.0;
        return false;
    }

    const Task& t = tasks[tid_to_drop];
    double prevPower = util_to_pow(slots[slot].gridUsedUtil);
    slots[slot].gridUsedUtil -= t.util;
    slots[slot].offlineGridScheduled.erase(tid_to_drop);
    tasks[tid_to_drop].is_sched = 0;
    double newPower = util_to_pow(slots[slot].gridUsedUtil);

    savedPower = prevPower - newPower;
    totalProfit -= t.profit;
    taskQueue.push({-t.util, t.id}); // min-util priority
    return true;
}

bool unsched_min_profit_task_solar(int slot, TaskQueue& taskQueue, double& savedPower) {
    if (slots[slot].offlineSolarScheduled.empty()) { 
        savedPower = 0.0; 
        return false; 
    }
    int tid_to_drop = -1;
    double min_profit = numeric_limits<double>::max();
    for (int tid : slots[slot].offlineSolarScheduled) {
        if (tasks[tid].profit < min_profit) {
            min_profit = tasks[tid].profit;
            tid_to_drop = tid;
        }
    }
    if (tid_to_drop == -1) { 
        savedPower = 0.0; 
        return false; 
    }

    const Task& t = tasks[tid_to_drop];
    double prevPower = util_to_pow(slots[slot].solarUsedUtil);
    slots[slot].solarUsedUtil -= t.util;
    slots[slot].offlineSolarScheduled.erase(tid_to_drop);
    tasks[tid_to_drop].is_sched = 0;
    double newPower = util_to_pow(slots[slot].solarUsedUtil);

    savedPower = prevPower - newPower; 
    totalProfit -= t.profit; 
    taskQueue.push({t.profit, t.id});
    return true;
}

/* ---------------------- Offline Scheduling Sub-routines ---------------------- */

void schedule_solar_pass(HeuristicMetric metric, set<int>& scheduled_task_ids) {
    for (int t = NUM_SLOTS - 1; t >= 0; t--) {
        TaskQueue available_tasks;
        for (const auto& task : tasks) {
            if (scheduled_task_ids.find(task.id) != scheduled_task_ids.end()) continue;
            if (task.arrival <= t && task.deadline >= t) {
                double priority = 0.0;
                switch (metric) {
                    case HPF:  priority = task.profit; break;
                    case HPDF: priority = (task.util > EPSILON) ? (task.profit / task.util) : 0.0; break;
                    case LPF:  priority = -task.profit; break;
                }
                available_tasks.push({priority, task.id});
            }
        }
        while (!available_tasks.empty()) {
            int tid = available_tasks.top().second;
            available_tasks.pop();
            if (slots[t].solarUsedUtil + tasks[tid].util <= slots[t].solarUtilLimit + EPSILON) {
                sched_task_solar(tid, t); 
                scheduled_task_ids.insert(tid);
            }
        }
    }
}

void schedule_grid_pass(TaskQueue& grid_candidates, set<int>& scheduled_task_ids, double gridCostMultiplier) {
    TaskQueue remaining_tasks;
    while (!grid_candidates.empty()) {
        int tid = grid_candidates.top().second;
        grid_candidates.pop();
        if (scheduled_task_ids.find(tid) != scheduled_task_ids.end()){
            continue;
        }
        const Task& task = tasks[tid];
        int best_t = -1;
        double min_marginal_cost = numeric_limits<double>::max();

        for (int t = task.arrival; t <= task.deadline; t++) {
            double prev = util_to_pow(slots[t].gridUsedUtil);
            double next = util_to_pow(slots[t].gridUsedUtil + task.util);
            double marginal_cost = get_tiered_cost(prev, next, slots[t].gridCostTiers) * gridCostMultiplier;
            if (marginal_cost < min_marginal_cost) {
                min_marginal_cost = marginal_cost;
                best_t = t;
            }
        }
        if (best_t != -1 && task.profit > min_marginal_cost) {
            sched_task_grid(tid, best_t);
            scheduled_task_ids.insert(tid);
        }
        else {
            remaining_tasks.push({task.profit, tid});
        }
    }
    grid_candidates = remaining_tasks;
}

/**
 * @brief Greedy Grid Scheduler.
 * Tries to execute the task in the current slot 't'.
 * No battery usage, no cost optimization lookahead.
 */
bool greedy_grid_schedule(int t, const Task& task, double currentGridUtil, double& costOut) {
    
    // 2. Calculate Cost (Pay as you go)
    double pPrev = util_to_pow(currentGridUtil);
    double pNext = util_to_pow(currentGridUtil + task.util);
    
    // Use the tiered cost helper we defined earlier
    costOut = get_tiered_cost(pPrev, pNext, slots[t].gridCostTiers);
    
    // 3. Profit Check
    // We'll keep a basic sanity check - on the higher side to not compromise SLA
    if (task.profit*SLA_LOSS_TOLERANCE < costOut) {
        return false; 
    }

    return true;
}

void offlinePhase2_FiniteBattery(TaskQueue& taskQueue, double batteryCaps) {
    const int MAX_ITERS = 2000;
    const double EPS = 1e-6;

    // batteryLevel[t] = stored power available after slot t
    batteryLevel.assign(NUM_SLOTS + 144, 0.0);

    int iteration = 0;
    while (++iteration <= MAX_ITERS) {
        double oldProfit = totalProfit;

        double totalPower = 0.0;
        for (int t = 0; t < NUM_SLOTS; t++) {
            totalPower += util_to_pow(slots[t].solarUsedUtil);
        }
        double pow_avg = totalPower / static_cast<double>(NUM_SLOTS);

        int t_max = -1;
        double max_power = -1.0;
        for (int t = 0; t < NUM_SLOTS; ++t) {
            double p = util_to_pow(slots[t].solarUsedUtil);
            if (p > max_power) {
                max_power = p;
                t_max = t;
            }
        }
        int t_min = -1;
        for (int t = t_max + 1; t < NUM_SLOTS; ++t) {
            if (util_to_pow(slots[t].solarUsedUtil) < pow_avg - EPS) {
                t_min = t;
                break;
            }
        }
        if (t_max == -1 || t_min == -1) {
            break;
        }

        double minAvailBattery = numeric_limits<double>::infinity();
        for (int j = t_max + 1; j <= t_min; ++j) {
            minAvailBattery = min(minAvailBattery, batteryCaps - batteryLevel[j]);
        }
        if (minAvailBattery <= EPS) {
            break;
        }

        double origPow_t_max = util_to_pow(slots[t_max].solarUsedUtil);
        double currPow_t_min = util_to_pow(slots[t_min].solarUsedUtil);
        double limit_from_t_max = max(0.0, origPow_t_max - pow_avg);
        double limit_for_t_min  = max(0.0, pow_avg - currPow_t_min);

        double transferable = min({minAvailBattery, limit_from_t_max, limit_for_t_min});
        if (transferable <= EPS) {
            break;
        }

        double savedAcc = 0.0;
        while (savedAcc + 1e-6 < transferable && !slots[t_max].offlineSolarScheduled.empty()) {
            double saved = 0.0;
            bool ok = unsched_min_profit_task_solar(t_max, taskQueue, saved);
            if (!ok || saved <= EPS) {
                break;
            }
            savedAcc += saved;
        }
        if (savedAcc <= EPS) {
            break;
        }

        double pow_to_move = min(savedAcc, transferable);
        for (int j = t_max + 1; j <= t_min; ++j) {
            batteryLevel[j] += pow_to_move;
            if (batteryLevel[j] > batteryCaps) {
                batteryLevel[j] = batteryCaps;
            }
        }

        double newPowerAtMin = util_to_pow(slots[t_min].solarUsedUtil) + pow_to_move;
        double newUtilAtMin  = pow_to_util(newPowerAtMin);
        double extraUtil     = max(0.0, newUtilAtMin - slots[t_min].solarUsedUtil);

        TaskQueue buffer;
        double power_consumed_at_tmin = 0.0;
        while (!taskQueue.empty() && extraUtil > EPS) {
            int tid = taskQueue.top().second;
            double prof = taskQueue.top().first;
            taskQueue.pop();
            const Task& task = tasks[tid];
            if (task.arrival > t_min || task.deadline < t_min) {
                buffer.push({prof, tid});
                continue;
            }
            double requiredPower = util_to_pow(slots[t_min].solarUsedUtil + task.util) - util_to_pow(slots[t_min].solarUsedUtil);
            if (requiredPower <= pow_to_move + EPS && slots[t_min].solarUsedUtil + task.util <= U_MAX + EPS) {
                sched_task_solar(tid, t_min);
                extraUtil -= requiredPower;
                power_consumed_at_tmin += requiredPower;
            } else {
                buffer.push({prof, tid});
            }
        }
        while (!buffer.empty()) {
            taskQueue.push(buffer.top());
            buffer.pop();
        }

        double consumption_remaining = power_consumed_at_tmin;
        if (consumption_remaining > EPS) {
            batteryLevel[t_min] = max(0.0, batteryLevel[t_min] - consumption_remaining);
            for (int j = t_min + 1; j < NUM_SLOTS; j++) {
                if (batteryLevel[j] > batteryLevel[j - 1]) {
                    batteryLevel[j] = batteryLevel[j - 1];
                }
            }
        }

        if (fabs(totalProfit - oldProfit) <= EPS) {
            break;
        }
    }
}

// Grid leveling: redistribute grid tasks to smooth power profile using lowest-util removals
void offline_grid_leveling(double gridCostMultiplier) {
    const int MAX_ITERS = 2000;
    const double EPS = 1e-6;
    TaskQueue taskQueue; // holds unscheduled grid tasks (priority: lowest util)

    int iteration = 0;
    while (++iteration <= MAX_ITERS) {
        double totalPower = 0.0;
        for (int t = 0; t < NUM_SLOTS; ++t) {
            totalPower += util_to_pow(slots[t].gridUsedUtil);
        }
        double pow_avg = totalPower / static_cast<double>(NUM_SLOTS);

        int t_max = -1;
        double max_power = -1.0;
        for (int t = 0; t < NUM_SLOTS; ++t) {
            double p = util_to_pow(slots[t].gridUsedUtil);
            if (p > max_power) {
                max_power = p;
                t_max = t;
            }
        }

        int t_min = -1;
        for (int t = t_max + 1; t < NUM_SLOTS; ++t) {
            if (util_to_pow(slots[t].gridUsedUtil) < pow_avg - EPS) {
                t_min = t;
                break;
            }
        }
        if (t_max == -1 || t_min == -1) {
            break;
        }

        double origPow_t_max = util_to_pow(slots[t_max].gridUsedUtil);
        double currPow_t_min = util_to_pow(slots[t_min].gridUsedUtil);
        double limit_from_t_max = max(0.0, origPow_t_max - pow_avg);
        double limit_for_t_min  = max(0.0, pow_avg - currPow_t_min);

        double transferable = min(limit_from_t_max, limit_for_t_min);
        if (transferable <= EPS) {
            break;
        }

        double savedAcc = 0.0;
        while (savedAcc + 1e-6 < transferable && !slots[t_max].offlineGridScheduled.empty()) {
            double saved = 0.0;
            bool ok = unsched_min_util_task_grid(t_max, taskQueue, saved);
            if (!ok || saved <= EPS) {
                break;
            }
            savedAcc += saved;
        }
        if (savedAcc <= EPS) {
            break;
        }

        double pow_to_move = min(savedAcc, transferable);

        // capacity gained at t_max is implicitly freed; add tasks to t_min respecting capacity and cost
        double newPowerAtMin = util_to_pow(slots[t_min].gridUsedUtil) + pow_to_move;
        double newUtilAtMin  = pow_to_util(newPowerAtMin);
        double extraUtil     = max(0.0, newUtilAtMin - slots[t_min].gridUsedUtil);

        TaskQueue buffer;
        while (!taskQueue.empty() && extraUtil > EPS) {
            int tid = taskQueue.top().second;
            taskQueue.pop();
            const Task& tk = tasks[tid];

            if (tk.arrival > t_min || tk.deadline < t_min) {
                buffer.push({-tk.util, tid});
                continue;
            }

            double reqPower = util_to_pow(slots[t_min].gridUsedUtil + tk.util) - util_to_pow(slots[t_min].gridUsedUtil);
            double cost = get_tiered_cost(util_to_pow(slots[t_min].gridUsedUtil), util_to_pow(slots[t_min].gridUsedUtil + tk.util), slots[t_min].gridCostTiers) * gridCostMultiplier;

            if (reqPower <= pow_to_move + EPS && tk.profit > cost) {
                sched_task_grid(tid, t_min);
                pow_to_move -= reqPower;
                extraUtil -= reqPower;
            } else {
                buffer.push({-tk.util, tid});
            }
        }
        while (!buffer.empty()) {
            taskQueue.push(buffer.top());
            buffer.pop();
        }

        if (pow_to_move <= EPS) {
            continue;
        }
    }

    // Any remaining unscheduled grid tasks: place them at lowest-cost feasible slot to avoid drops
    while (!taskQueue.empty()) {
        int tid = taskQueue.top().second;
        taskQueue.pop();
        const Task& tk = tasks[tid];
        int best_t = -1;
        double best_cost = numeric_limits<double>::max();
        for (int t = tk.arrival; t <= tk.deadline; t++) {
            double c = get_tiered_cost(util_to_pow(slots[t].gridUsedUtil), util_to_pow(slots[t].gridUsedUtil + tk.util), slots[t].gridCostTiers) * gridCostMultiplier;
            if (c < best_cost) {
                best_cost = c;
                best_t = t;
            }
        }
        if (best_t != -1 && tk.profit > best_cost) {
            sched_task_grid(tid, best_t);
        }
    }

    // persist planned grid power usage for online phase
    for (int t = 0; t < NUM_SLOTS; ++t) {
        offlineGridUsage[t] = util_to_pow(slots[t].gridUsedUtil);
    }
}

void reset_simulation_state() {
    totalProfit = 0.0;
    totalGridCost = 0.0;
    batteryLevel.assign(NUM_SLOTS + 144, 0.0);
    offlineGridUsage.assign(NUM_SLOTS + 144, 0.0);
    for (auto &t : tasks) {
        t.is_sched = 0;
    }
    for (int t = 0; t < NUM_SLOTS; ++t) {
        slots[t].solarUsedUtil = 0.0;
        slots[t].gridUsedUtil = 0.0;
        slots[t].offlineSolarScheduled.clear();
        slots[t].offlineGridScheduled.clear();
        slots[t].sdcGridBatteryLevel = 0.0;
    }
}

void run_offline_scheduler(GridHeuristic heuristic, BatteryMode mode, double gridCostMultiplier, double batteryCaps) {
    set<int> scheduled_task_ids;
    TaskQueue task_queue_A, task_queue_B, unscheduled_tasks; 

    // If using SDC, we allocate partial capacity to Solar if mixing modes, 
    // but standard offline runs usually assume full capacity for comparison.
    // For Experiment consistency, we use full cap unless specified.
    
    for(const auto& task : tasks) task_queue_A.push({task.profit, task.id});

    switch (heuristic) {
        case APP_A_SOLAR_ONLY_FINITE:
        case APP_C_HYBRID_GREEDY:
        case APP_D_HYBRID_OPTIMAL:
        case APP_E_HYBRID_SDC_LIMITED:
        case APP_F_HYBRID_SDC_OPTIMAL:
            schedule_solar_pass(HPF, scheduled_task_ids);
            for(const auto& task : tasks) {
                if(scheduled_task_ids.find(task.id) == scheduled_task_ids.end()) {
                    unscheduled_tasks.push({task.profit, task.id});
                }
            }
            break;
        case APP_B_GRID_ONLY_FINITE:
            schedule_grid_pass(task_queue_A, scheduled_task_ids, gridCostMultiplier);
            for(const auto& task : tasks) {
                if(scheduled_task_ids.find(task.id) == scheduled_task_ids.end()) {
                    unscheduled_tasks.push({task.profit, task.id});
                }
            }
            break;
    }

    if (mode == FINITE_BATTERY) {
        offlinePhase2_FiniteBattery(unscheduled_tasks, batteryCaps);
    }     
    if (heuristic != APP_A_SOLAR_ONLY_FINITE) {
        schedule_grid_pass(unscheduled_tasks, scheduled_task_ids, gridCostMultiplier);
        if (heuristic == APP_D_HYBRID_OPTIMAL) {
            // only approach D uses optimal offline grid scheduling, rest are all online
            // Smooth grid usage offline and record planned power profile
            offline_grid_leveling(gridCostMultiplier);
        }
    }


}

/* ---------------------- Modular SDC Logic ---------------------- */

struct SDCDecision {
    bool executed;
    double gridCostIncurred;
    double batteryConsumed;
};

/**
 * @brief Logic based on Look-Behind Leveling.
 * Uses a moving average of history to define the "Leveling Target".
 */
SDCDecision attempt_sdc_grid_schedule(
    int t, 
    const Task& task, 
    double currentGridUtil, 
    double currentBattery, 
    double P_target, 
    const vector<double>& gridCosts
) {
    SDCDecision res = {false, 0.0, 0.0};

    double pPrev = util_to_pow(currentGridUtil);
    double pNext = util_to_pow(currentGridUtil + task.util);
    double powerReq = pNext - pPrev; 

    // 2. Look-Behind Leveling Logic
    // If running this task puts us above our historical average (P_target),
    // we use the battery to try and "level" the draw back down to P_target.
    double battery_support = 0.0;
    
    if (pNext > P_target && currentBattery > EPSILON) {
        double needed = pNext - P_target;
        battery_support = min({needed, currentBattery, powerReq});
    }

    // 3. Marginal Cost Check
    double grid_draw_increase = max(0.0, powerReq - battery_support);
    double cost = get_tiered_cost(pPrev, pPrev + grid_draw_increase, gridCosts);

    // 4. Execution Decision
    bool isCritical = (task.deadline <= t);
    bool isEconomical = (task.profit * SLA_LOSS_TOLERANCE >= cost);

    if (isEconomical || isCritical) {
        res.executed = true;
        res.gridCostIncurred = cost;
        res.batteryConsumed = battery_support;
    }

    return res;
}

SimResult run_SDC_Online_Shaving(
    const vector<vector<int>>& arrivalsActualBySlot,
    const vector<Task>& tasks_actual,
    double fullBatteryCapacity
) {
    reset_simulation_state();

    double gridBattCap = fullBatteryCapacity * GRID_BATT_RATIO;
    double currentGridBattery = 0.0; 
    
    // Trackers for Sell-Back Credit (to prevent large battery cost spikes)
    double totalUnitsPurchased = 0.0;
    double totalSpendOnCharging = 0.0;
    
    // History for Look-Behind window
    vector<double> powerHistory(NUM_SLOTS + 144, 0.0);
    set<int> executedTaskIds;
    TaskQueue pendingTasks; 

    for (int t = 0; t < NUM_SLOTS; ++t) {
        // A. Ingest Arrivals
        for (int tid : arrivalsActualBySlot[t]) {
            pendingTasks.push({tasks_actual[tid].profit, tid});
        }

        // B. Calculate Look-Behind Target (P_target)
        // Average of the last SDC_WINDOW slots. 
        // We default to TIER1_LIMIT_W if history is empty.
        double historySum = 0;
        int count = 0;
        for (int i = max(0, t - SDC_WINDOW); i < t; ++i) {
            historySum += powerHistory[i];
            count++;
        }
        double P_target = (count > 0) ? (historySum / count) : TIER1_LIMIT_W;
        
        // C. Process Tasks
        vector<int> deferred;
        while (!pendingTasks.empty()) {
            int tid = pendingTasks.top().second;
            pendingTasks.pop();
            const Task& task = tasks_actual[tid];

            if (t > task.deadline || executedTaskIds.count(tid)) {
                continue;
            }

            SDCDecision decision = attempt_sdc_grid_schedule(
                t, task, slots[t].gridUsedUtil, currentGridBattery, 
                P_target, slots[t].gridCostTiers
            );

            if (decision.executed) {
                executedTaskIds.insert(tid);
                totalGridCost += decision.gridCostIncurred;
                totalProfit += task.profit;
                slots[t].gridUsedUtil += task.util; 
                currentGridBattery -= decision.batteryConsumed;
            } else {
                deferred.push_back(tid);
            }
        }
        for (int tid : deferred) {
            pendingTasks.push({tasks_actual[tid].profit, tid});
        }

        // D. Valley Filling (Air-tight Recharge)
        // We charge if we are below P_target AND below Tier 1 limit.
        double pActual = util_to_pow(slots[t].gridUsedUtil);
        double rechargeLimit = min(P_target, TIER1_LIMIT_W);
        
        if (pActual < rechargeLimit) {
            double headroom = rechargeLimit - pActual;
            double space = gridBattCap - currentGridBattery;
            double chargeAmt = min(headroom, space);
            
            if (chargeAmt > 0) {
                double chargeCost = get_tiered_cost(pActual, pActual + chargeAmt, slots[t].gridCostTiers);
                totalGridCost += chargeCost;
                currentGridBattery += chargeAmt * BATTERY_CHARGE_EFF;
                
                // Track for refund logic
                totalUnitsPurchased += chargeAmt;
                totalSpendOnCharging += chargeCost;
            }
        }
        
        // Record history for the next slot's look-behind
        powerHistory[t] = util_to_pow(slots[t].gridUsedUtil);
        slots[t].sdcGridBatteryLevel = currentGridBattery;
    }

    // E. Final Energy Credit (The "Sell-Back")
    // This prevents large batteries from showing artificial cost spikes.
    if (totalUnitsPurchased > 0 && currentGridBattery > 0) {
        double avgPrice = totalSpendOnCharging / totalUnitsPurchased;
        double refund = (currentGridBattery / BATTERY_CHARGE_EFF) * avgPrice;
        totalGridCost -= refund;
    }

    SimResult res;
    res.tasksOnGrid = executedTaskIds.size();
    res.tasksDropped = tasks_actual.size() - executedTaskIds.size();
    res.netRevenue = totalGridCost; 
    return res;
}
/* ---------------------- Online Adjustment (Algo 5) ---------------------- */

SimResult online_adjust_universal(
    ExperimentConfig config,
    const vector<double>& actualSolar,
    const vector<vector<int>>& arrivalsActualBySlot,
    const vector<Task>& tasks_actual,
    double totalBatteryCapacity
) {
    // --- 1. PRE-CALCULATION (O(Total Scheduled Tasks)) ---
    // Map Task ID -> The specific slot it is scheduled for solar (-1 if none)
    // This replaces the O(S) "futureT" loop with a O(1) lookup.
    vector<int> taskScheduledSolarSlot(tasks_actual.size(), -1);
    for (int t = 0; t < NUM_SLOTS; t++) {
        for (int tid : slots[t].offlineSolarScheduled) {
            if (tid < 0 || tid >= (int)tasks_actual.size()) {
                continue; 
            }
            if (tid < (int)taskScheduledSolarSlot.size()) {
                taskScheduledSolarSlot[tid] = t;
            }
        }
    }

    // --- 2. SETUP & STATE ---
    double solarBatteryCap = totalBatteryCapacity * config.solarBatteryRatio;
    double gridBatteryCap  = totalBatteryCapacity * config.sdcBatteryRatio;
    double currentSolarBattery = 0.0; 
    double currentGridBattery  = 0.0; 

    double totalProfit = 0.0, totalGridCost = 0.0;
    int solarCount = 0, gridCount = 0, arrivedCount = 0;
    double totalGridUnitsBought = 0.0, totalGridSpend = 0.0;

    // Use vector<bool> for O(1) execution checks
    vector<bool> executed(tasks_actual.size(), false);
    vector<double> gridPowerHistory(NUM_SLOTS, 0.0);
    vector<double> solarPowerHistory(NUM_SLOTS, 0.0);
    vector<double> solarBatteryHistory(NUM_SLOTS, 0.0);
    vector<double> gridUsageHistory(NUM_SLOTS, 0.0);

    // Persistent Grid Backlog: Tasks stay here across slots until executed or expired.
    // Stores {profit, taskId} to prioritize high-value tasks.
    priority_queue<pair<double, int>> gridBacklog;

    for (int t = 0; t < NUM_SLOTS; t++) {
        
        // --- PHASE 0: NEW ARRIVALS ---
        for (int tid : arrivalsActualBySlot[t]) {
            arrivedCount++;
            // If the task has no solar plan, it immediately becomes a grid candidate.
            if (taskScheduledSolarSlot[tid] == -1) {
                gridBacklog.push({tasks_actual[tid].profit, tid});
            }
        }

        // --- PHASE A: PLANNED SOLAR (Priority 1) ---
        double solarAvail = actualSolar[t] + currentSolarBattery;
        double actualSolarUsedUtil = 0.0;

        // Process tasks planned for THIS specific slot
        for (int tid : slots[t].offlineSolarScheduled) {
            if (tid < 0 || tid >= (int)tasks_actual.size()) {
                continue; 
            }
            const Task& task = tasks_actual[tid];
            
            double pNext = util_to_pow(actualSolarUsedUtil + task.util);
            double pReq = pNext - util_to_pow(actualSolarUsedUtil);

            if (pReq <= solarAvail + EPSILON && actualSolarUsedUtil + task.util <= U_MAX + EPSILON) {
                actualSolarUsedUtil += task.util;
                solarAvail -= pReq;
                totalProfit += task.profit;
                executed[tid] = true;
                solarCount++;
            } 
            else {
                // FAILED SOLAR: Task is now eligible for Grid Backlog
                gridBacklog.push({task.profit, tid});
            }
        }

        // --- PHASE B: GRID & SCAVENGING (Priority 2) ---
        double actualGridUsedUtil = 0.0;
        double batteryDischargedThisSlot = 0.0;
        vector<pair<double, int>> deferred; // To hold tasks for the next slot

        while (!gridBacklog.empty()) {
            int tid = gridBacklog.top().second;
            gridBacklog.pop();
            const Task& task = tasks_actual[tid];

            // Lazy pruning
            if (executed[tid]) {
                continue;
            }
            if (t > task.deadline) {
                continue;
            }
            if (config.solarHeuristic != APP_B_GRID_ONLY_FINITE) {
                // 1. SCAVENGE LEFTOVER SOLAR (Highest Priority Grid Strategy)
                double pNextS = util_to_pow(actualSolarUsedUtil + task.util);
                double pReqS = pNextS - util_to_pow(actualSolarUsedUtil);
                if (pReqS <= solarAvail + EPSILON && actualSolarUsedUtil + task.util <= U_MAX + EPSILON) {
                    actualSolarUsedUtil += task.util;
                    solarAvail -= pReqS;
                    totalProfit += task.profit;
                    executed[tid] = true;
                    solarCount++;
                    continue; 
                }
            }
            // 2. GRID EXECUTION
            bool runNow = false;
            double cost = 0.0;
            double pPrevG = util_to_pow(actualGridUsedUtil);
            double pNextG = util_to_pow(actualGridUsedUtil + task.util);

            if (config.gridStrategy == GRID_GREEDY) {
                
                cost = get_tiered_cost(pPrevG, pNextG, slots[t].gridCostTiers);
                if (task.profit * SLA_LOSS_TOLERANCE >= cost) {
                    runNow = true;
                }
                
            } 
            else if (config.gridStrategy == GRID_OPTIMAL_OFFLINE) {
                double plannedP = offlineGridUsage[t];
                bool withinOfflineTier = (pNextG <= plannedP + EPSILON) || (!crosses_tier(pPrevG, pNextG));
                
                if ((withinOfflineTier || t == task.deadline)) {
                    cost = get_tiered_cost(pPrevG, pNextG, slots[t].gridCostTiers);
                    if (task.profit * SLA_LOSS_TOLERANCE >= cost) {
                        runNow = true;
                    }
                }
            }
            else if (config.gridStrategy == GRID_SDC_SHAVING) {
                // Calculate P_target from history
                double historySum = 0; int count = 0;
                for (int i = max(0, t - SDC_WINDOW); i < t; ++i) { 
                    historySum += gridPowerHistory[i]; 
                    count++; 
                }
                double P_target;
                if(count > 0) {
                    P_target = (historySum / count);
                }
                else {
                    P_target = TIER1_LIMIT_W;
                }
                SDCDecision dec = attempt_sdc_grid_schedule(t, task, actualGridUsedUtil, currentGridBattery, P_target, slots[t].gridCostTiers);
                if (dec.executed) {
                    runNow = true;
                    cost = dec.gridCostIncurred;
                    currentGridBattery -= dec.batteryConsumed;
                    batteryDischargedThisSlot += dec.batteryConsumed;
                }
            }

            if (runNow) {
                actualGridUsedUtil += task.util;
                totalGridCost += cost;
                totalProfit += task.profit;
                executed[tid] = true;
                gridCount++;
            } 
            else if (t < task.deadline) {
                // Not executed but not expired: defer to next slot
                deferred.push_back({task.profit, tid});
            }
        }

        // Re-insert deferred tasks for the next iteration
        for (auto& d : deferred) {
            gridBacklog.push(d);
        }

        // --- PHASE C: BATTERY UPDATES ---
        double solarConsumedP = util_to_pow(actualSolarUsedUtil);
        if (actualSolar[t] > solarConsumedP) {
            double surplus = actualSolar[t] - solarConsumedP;
            solarPowerHistory[t] = solarConsumedP;
            currentSolarBattery = min(solarBatteryCap, currentSolarBattery + surplus * BATTERY_CHARGE_EFF);
        } 
        else {
            double deficit = solarConsumedP - actualSolar[t];
            solarPowerHistory[t] = actualSolar[t];
            solarBatteryHistory[t] = deficit;
            currentSolarBattery = max(0.0, currentSolarBattery*BATTERY_DISCHARGE_EFF - deficit);
        }

        // Track actual grid power draw (accounting for battery)
        double actualGridPowerDraw = util_to_pow(actualGridUsedUtil);
        if (config.gridStrategy == GRID_SDC_SHAVING) {
            actualGridPowerDraw -= batteryDischargedThisSlot;
            double gridP = actualGridPowerDraw;
            if (config.solarHeuristic == APP_F_HYBRID_SDC_OPTIMAL) {
                currentGridBattery = gridBatteryCap;
            } 
            else {
                double target = TIER1_LIMIT_W * 0.9;
                if (gridP < target) {
                    double charge = min(target - gridP, gridBatteryCap - currentGridBattery);

                    if (charge > 0) {
                        double cCost = get_tiered_cost(gridP, gridP + charge, slots[t].gridCostTiers);
                        totalGridCost += cCost;
                        currentGridBattery += charge * BATTERY_CHARGE_EFF;
                        totalGridUnitsBought += charge; 
                        totalGridSpend += cCost;
                        actualGridPowerDraw += charge;
                    }
                }
            }
        }
        gridUsageHistory[t] = actualGridPowerDraw;
        gridPowerHistory[t] = util_to_pow(actualGridUsedUtil);
    }

    // --- PHASE D: FINALIZATION ---
    if (totalGridUnitsBought > 0 && currentGridBattery > 0) {
        totalGridCost -= (currentGridBattery / BATTERY_DISCHARGE_EFF) * (totalGridSpend / totalGridUnitsBought);
        gridUsageHistory[NUM_SLOTS - 1] -= currentGridBattery;
    }

    SimResult res;
    res.netRevenue = totalGridCost;
    res.tasksOnSolar = solarCount; 
    res.tasksOnGrid = gridCount;
    res.tasksDropped = arrivedCount - (solarCount + gridCount);
    res.slotWiseGridUsage = gridUsageHistory;
    res.slotWiseSolarUsage = solarPowerHistory;
    res.slotWiseSolarBatteryLevel = solarBatteryHistory;
    return res;
}

class TestSuite {
public:
    /* ---------------- Configuration ---------------- */
    static int counter;
    vector<double> battery_caps = {0, 500, 1000, 5000, 10000, 20000, 30000, 40000, 50000, 75000};
    vector<double> sla_values   = {1.0, 5.0, 10.0, 100.0, 500.0};
    vector<double> eff_values   = {0.5, 0.75, 0.85, 0.9, 0.99};
    vector<double> rho_values   = {0.03, 0.05, 0.1, 0.15, 0.18};

    const double DEFAULT_GRID_MULT = 1.0;
    const double DEFAULT_BATT_CAP  = 10000.0;
    string task_file;
    string power_file;
    TestSuite(
        const string& taskFile,
        const string& powerFile
    ) : task_file(taskFile), power_file(powerFile) { 
        counter++; 
        cout << "Initialized TestSuite #" << counter << "\n";
        cout << "Task File: " << task_file << "\n";
        cout << "Power File: " << power_file << "\n";
    }
    struct AlgoConfig {
        string name;
        GridHeuristic solarHeuristic;
        GridStrategy gridStrategy;
        double solarBattRatio;
        double gridBattRatio;
    };

    vector<AlgoConfig> algos = {
        {"PURE_SOLAR", APP_A_SOLAR_ONLY_FINITE, GRID_NONE, 0.5, 0.5},
        {"GRID_GREEDY", APP_C_HYBRID_GREEDY, GRID_GREEDY, 0.5, 0.5},
        {"GRID_OPTIMAL", APP_D_HYBRID_OPTIMAL, GRID_OPTIMAL_OFFLINE, 0.5, 0.5},
        {"SDC_ONLY", APP_B_GRID_ONLY_FINITE, GRID_SDC_SHAVING, 0.5, 0.5},
        {"SOLAR_SDC", APP_E_HYBRID_SDC_LIMITED, GRID_SDC_SHAVING, 0.5, 0.5},
        {"SDC_OPTIMAL", APP_F_HYBRID_SDC_OPTIMAL, GRID_SDC_SHAVING, 0.5, 0.5}
    };

    /* ---------------- Utilities ---------------- */

    void write_header(ofstream& out) {
        out << "Benchmark,Value,Algorithm,"
            << "TasksSolar,TasksGrid,TasksDropped,NetRevenue\n";
    }

    void write_row(ofstream& out,
                   const string& bench,
                   double value,
                   const string& algo,
                   const SimResult& r) {
        out << bench << "," << value << "," << algo << ","
            << r.tasksOnSolar << ","
            << r.tasksOnGrid << ","
            << r.tasksDropped << ","
            << r.netRevenue << "\n";
    }

    void load_actual(double rho,
                     vector<double>& actualSolar,
                     vector<Task>& tasks_actual,
                     vector<vector<int>>& arrivals) {
        string rho_str = to_string(rho).substr(0, 4);
        string suffix  = "_5_percent_deviation";

        actualSolar = FileReader::readActualPower(power_file);

        FileReader::readActualTasks(task_file, tasks_actual, arrivals);
    }

    SimResult run_algo(const AlgoConfig& a,
                       const vector<double>& actualSolar,
                       const vector<vector<int>>& arrivals,
                       const vector<Task>& tasks_actual,
                       double battCap) {

        reset_simulation_state();

        BatteryMode mode =
            (battCap <= EPSILON) ? NO_BATTERY : FINITE_BATTERY;

        run_offline_scheduler(
            a.solarHeuristic,
            mode,
            DEFAULT_GRID_MULT,
            battCap*a.solarBattRatio
        );

        ExperimentConfig cfg{
            a.name,
            a.solarHeuristic,
            a.gridStrategy,
            a.solarBattRatio,
            a.gridBattRatio
        };

        return online_adjust_universal(
            cfg, actualSolar, arrivals, tasks_actual, battCap
        );
    }

    /* ---------------- Benchmark A: Battery ---------------- */

    void bench_battery() {
        string filename = "benchmark_battery" + to_string(counter) + ".csv";
        ofstream out(filename);
        write_header(out);
        cout << "Running Battery Capacity Benchmark...\n";
        auto start = chrono::high_resolution_clock::now();
        vector<double> actualSolar;
        vector<Task> tasks_actual;
        vector<vector<int>> arrivals;
        load_actual(0.03, actualSolar, tasks_actual, arrivals);

        for (double cap : battery_caps) {
            batteryCapacity = cap;

            for (auto& a : algos) {
                reset_simulation_state();
                SimResult r =
                    run_algo(a, actualSolar, arrivals, tasks_actual, cap);
                write_row(out, "Battery", cap, a.name, r);
            }
        }
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        cout << "Battery Capacity Benchmark completed in " << duration.count() << " seconds.\n";  
    }

    /* ---------------- Benchmark B: SLA ---------------- */

    void bench_sla() {
        string filename = "benchmark_sla" + to_string(counter) + ".csv";
        ofstream out(filename);
        write_header(out);
        cout << "Running SLA Benchmark...\n";
        auto start = chrono::high_resolution_clock::now();
        vector<double> actualSolar;
        vector<Task> tasks_actual;
        vector<vector<int>> arrivals;
        load_actual(0.03, actualSolar, tasks_actual, arrivals);

        for (double sla : sla_values) {
            SLA_LOSS_TOLERANCE = sla;   // must not be constexpr

            for (auto& a : algos) {
                SimResult r =
                    run_algo(a, actualSolar, arrivals, tasks_actual, DEFAULT_BATT_CAP);
                write_row(out, "SLA", sla, a.name, r);
            }
        }
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        cout << "SLA Benchmark completed in " << duration.count() << " seconds.\n";
    }

    /* ---------------- Benchmark C: Efficiency ---------------- */

    void bench_efficiency() {
        string filename = "benchmark_efficiency" + to_string(counter) + ".csv";
        ofstream out(filename);
        write_header(out);
        cout << "Running Efficiency Benchmark...\n";
        auto start = chrono::high_resolution_clock::now();
        vector<double> actualSolar;
        vector<Task> tasks_actual;
        vector<vector<int>> arrivals;
        load_actual(0.03, actualSolar, tasks_actual, arrivals);

        for (double eff : eff_values) {
            BATTERY_CHARGE_EFF    = eff;
            BATTERY_DISCHARGE_EFF = eff;

            for (auto& a : algos) {
                SimResult r =
                    run_algo(a, actualSolar, arrivals, tasks_actual, DEFAULT_BATT_CAP);
                write_row(out, "Efficiency", eff, a.name, r);
            }
        }
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        cout << "Efficiency Benchmark completed in " << duration.count() << " seconds.\n";
    }

    void log_grid(const string& logFilename) {
        cout << "Logging grid usage to " << logFilename << "...\n";
        auto start = chrono::high_resolution_clock::now();
        ofstream out(logFilename);
        // Header: Hour, then Power and Cost columns for each requested algorithm
        out << "Hour,GridGreedy_Power,GridGreedy_Cost,GridOptimal_Power,GridOptimal_Cost,SolarSDC_Power,SolarSDC_Cost\n";
        BATTERY_CHARGE_EFF = 0.999;
        BATTERY_DISCHARGE_EFF = 0.999;
        SLA_LOSS_TOLERANCE = 1000000;
        // Define the subset of algorithms requested
        vector<AlgoConfig> logAlgos = {
            {"GRID_GREEDY", APP_C_HYBRID_GREEDY, GRID_GREEDY, 0.5, 0.5},
            {"GRID_OPTIMAL", APP_D_HYBRID_OPTIMAL, GRID_OPTIMAL_OFFLINE, 0.5, 0.5},
            {"SOLAR_SDC", APP_E_HYBRID_SDC_LIMITED, GRID_SDC_SHAVING, 0.5, 0.5}
        };
        vector<double> actualSolar;
        vector<Task> tasks_actual;
        vector<vector<int>> arrivals;
        load_actual(0.03, actualSolar, tasks_actual, arrivals);
        // Matrix to store hourly results [Hour][Algo_Index * 2 (Power, Cost)]
        int totalHours = NUM_SLOTS / 30;
        vector<vector<double>> hourlyData(totalHours, vector<double>(6, 0.0));

        for (int i = 0; i < logAlgos.size(); i++) {
            
            SimResult r = run_algo(logAlgos[i], actualSolar, arrivals, tasks_actual, 50000.0);

            // 2. Aggregate slot data into hours
            for (int t = 0; t < NUM_SLOTS; ++t) {
                int hour = t / 30;
                if (hour >= totalHours) {
                    break;
                }

                double pGrid = r.slotWiseGridUsage[t];
                // Calculate cost for this specific slot using the tiered cost function
                double costGrid = get_tiered_cost(0, pGrid, slots[t].gridCostTiers);

                hourlyData[hour][i * 2] += pGrid;     // Total Power in Watt-slots for the hour
                hourlyData[hour][i * 2 + 1] += costGrid; // Total Cost in INR for the hour
            }
        }

        

        // 3. Write aggregated data to CSV
        for (int h = 0; h < totalHours; ++h) {
            out << h + 1; // Hour index
            for (int j = 0; j < 6; ++j) {
                out << "," << hourlyData[h][j];
            }
            out << "\n";
        }
        out.close();
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        cout << "[Success] Grid usage logged to " << logFilename << " in " << duration.count() << " seconds.\n";
    }
    void log_usage(){
        if(counter == 2){
            cout << "Logger activated...\n";
            auto start = chrono::high_resolution_clock::now();
            ofstream out1("detailed_log_optimal.csv");
            out1 << "Timeslot,GridUsage,SolarUsage,SolarBatteryUsage\n";
            vector<double> actualSolar;
            vector<Task> tasks_actual;
            vector<vector<int>> arrivals;
            BATTERY_CHARGE_EFF = 0.999;
            BATTERY_DISCHARGE_EFF = 0.999;
            SLA_LOSS_TOLERANCE = 1000000;
            load_actual(0.03, actualSolar, tasks_actual, arrivals);
            SimResult r = run_algo(algos[2], actualSolar, arrivals, tasks_actual, 50000.0);
            for(int t = 0; t < NUM_SLOTS; t++){
                out1 << t+1 << "," << r.slotWiseGridUsage[t] << "," << r.slotWiseSolarUsage[t] << "," << r.slotWiseSolarBatteryLevel[t] << "\n";
            }
            out1.close();
            ofstream out2("detailed_log_sdc.csv");
            out2 << "Timeslot,GridUsage,SolarUsage,SolarBatteryUsage\n";
            r = run_algo(algos[4], actualSolar, arrivals, tasks_actual, 50000.0);
            for(int t = 0; t < NUM_SLOTS; t++){
                out2 << t+1 << "," << r.slotWiseGridUsage[t] << "," << r.slotWiseSolarUsage[t] << "," << r.slotWiseSolarBatteryLevel[t] << "\n";
            }
            out2.close();
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> duration = end - start;
            cout << "[Success] Detailed usage logs created in " << duration.count() << " seconds.\n";
            ofstream out3("detailed_log_greedy.csv");
            out3 << "Timeslot,GridUsage,SolarUsage,SolarBatteryUsage\n";
            r = run_algo(algos[1], actualSolar, arrivals, tasks_actual, 10000.0);
            for(int t = 0; t < NUM_SLOTS; t++){
                out3 << t+1 << "," << r.slotWiseGridUsage[t] << "," << r.slotWiseSolarUsage[t] << "," << r.slotWiseSolarBatteryLevel[t] << "\n";
            }
        }
        else{
            return;
        }
    }
    /* ---------------- Public Entry ---------------- */

    void run_all_tests() {
        cout << "Running benchmarks...\n";
        bench_battery();
        // bench_sla();
        // bench_efficiency();
        // log_grid("grid_usage_log_" + to_string(counter) + ".csv");
        // log_usage();
        cout << "All benchmarks completed.\n";
    }
};

int TestSuite::counter = 0;
int main(){
    FileReader::readPower("power_reallife.csv");
    FileReader::readGridCost("grid_cost.csv");
    FileReader::readTasks("task_reallife.csv");
    cout << "Running test suite for scenario 1...\n";
    TestSuite ts0("task_reallife.csv", "power_reallife.csv");
    ts0.run_all_tests();
    cout << "\n\n";
    cout << "Running test suite for scenario 1...\n";
    TestSuite ts1("tasks_5_deviation.csv", "results_5_power_deviation.csv");
    ts1.run_all_tests();
    cout << "\n\n";
    cout << "Running test suite for scenario 2...\n";
    TestSuite ts2("tasks_10_deviation.csv", "results_10_power_deviation.csv");
    ts2.run_all_tests();
    cout << "\n\n";
    cout << "Running test suite for scenario 3...\n";   
    TestSuite ts3("tasks_20_deviation.csv", "results_20_power_deviation.csv");
    ts3.run_all_tests();
    cout << "\n\n";
    cout << "Running test suite for scenario 4...\n";
    TestSuite ts4("results_10_robust_tasks.csv", "results_10_robust_power.csv");
    ts4.run_all_tests();
    cout << "\n";
    return 0;
}

