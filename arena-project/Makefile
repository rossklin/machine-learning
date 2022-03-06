CC=g++
CPPFLAGS=--std=c++17 -fopenmp
LDFLAGS=-lnlopt
SOURCES=agent.cpp choice.cpp game_generator.cpp pod_game.cpp pod_game_generator.cpp evaluator.cpp team_evaluator.cpp simple_pod_evaluator.cpp tree_evaluator.cpp arena.cpp game.cpp pod_agent.cpp population_manager.cpp random_tournament.cpp utility.cpp
SRC_DIR=./src
BUILD_DIR=./build
SRC_PATHS=$(SOURCES:%=$(SRC_DIR)/%)
DBG_DIR=./debug_build
OBJ=$(SRC_PATHS:%.cpp=$(BUILD_DIR)/%.o)
DBG_OBJ=$(SRC_PATHS:%.cpp=$(DBG_DIR)/%.o)
DEP = $(OBJ:%.o=%.d) $(DBG_OBJ:%.o=%.d)

# ifdef DEBUG
# CPPFLAGS=--std=c++17 -fopenmp -ggdb
# else 
# CPPFLAGS=--std=c++17 -fopenmp -O3
# endif

# Disable default rules
.SUFFIXES:

default: pure_train run_arena

pure_train : $(BUILD_DIR)/pure_train $(DBG_DIR)/pure_train 
	rm pure_train || true
	ln -s build/pure_train

$(BUILD_DIR)/pure_train : $(OBJ) $(SRC_DIR)/pure_train.cpp
	# Create build directories - same structure as sources.
	mkdir -p $(@D)
	# Just link all the object files.
	$(CC) $(CPPFLAGS) -O3 $^ -o $@ $(LDFLAGS)

$(DBG_DIR)/pure_train : $(DBG_OBJ) $(SRC_DIR)/pure_train.cpp
	# Create build directories - same structure as sources.
	mkdir -p $(@D)
	# Just link all the object files.
	$(CC) -DDEBUG $(CPPFLAGS) -ggdb $^ -o $@ $(LDFLAGS)

run_arena : $(BUILD_DIR)/run_arena $(DBG_DIR)/run_arena
	rm run_arena || true
	ln -s build/run_arena

$(BUILD_DIR)/run_arena : $(OBJ) $(SRC_DIR)/run_arena.cpp
	# Create build directories - same structure as sources.
	mkdir -p $(@D)
	# Just link all the object files.
	$(CC) $(CPPFLAGS) -O3 $^ -o $@ $(LDFLAGS)

$(DBG_DIR)/run_arena : $(DBG_OBJ) $(SRC_DIR)/run_arena.cpp
	# Create build directories - same structure as sources.
	mkdir -p $(@D)
	# Just link all the object files.
	$(CC) -DDEBUG $(CPPFLAGS) -ggdb $^ -o $@ $(LDFLAGS)

-include $(DEP)

# Build target for every single object file.
# The potential dependency on header files is covered
# by calling `-include $(DEP)`.
$(BUILD_DIR)/%.o : %.cpp
	mkdir -p $(@D)
	# The -MMD flags additionaly creates a .d file with
	# the same name as the .o file.
	$(CC) $(CPPFLAGS) -O3 -MMD -c $< -o $@ $(LDFLAGS)

$(DBG_DIR)/%.o : %.cpp
	mkdir -p $(@D)
	# The -MMD flags additionaly creates a .d file with
	# the same name as the .o file.
	$(CC) -DDEBUG $(CPPFLAGS) -ggdb -MMD -c $< -o $@ $(LDFLAGS)

.PHONY : clean

clear_data:
	rm -rf data/* save/* brains/* || true

clean :
	# This should remove all generated files.
	rm -rf {.,$(BUILD_DIR),$(DBG_DIR)}/{pure_train,run_arena} $(OBJ) $(DBG_OBJ) $(DEP) pod_codingame.cpp || true

