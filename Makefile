CC=g++
CPPFLAGS=--std=c++17 -fopenmp
SOURCES=agent.cpp choice.cpp game_generator.cpp pod_game.cpp pod_game_generator.cpp evaluator.cpp team_evaluator.cpp simple_pod_evaluator.cpp tree_evaluator.cpp arena.cpp game.cpp pod_agent.cpp population_manager.cpp random_tournament.cpp utility.cpp
BUILD_DIR=./build
DBG_DIR=./debug_build
OBJ=$(SOURCES:%.cpp=$(BUILD_DIR)/%.o)
DBG_OBJ=$(SOURCES:%.cpp=$(DBG_DIR)/%.o)
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

$(BUILD_DIR)/pure_train : $(OBJ) pure_train.cpp
	# Create build directories - same structure as sources.
	mkdir -p $(@D)
	# Just link all the object files.
	$(CC) $(CPPFLAGS) -O3 $^ -o $@

$(DBG_DIR)/pure_train : $(DBG_OBJ) pure_train.cpp
	# Create build directories - same structure as sources.
	mkdir -p $(@D)
	# Just link all the object files.
	$(CC) $(CPPFLAGS) -ggdb $^ -o $@

run_arena : $(BUILD_DIR)/run_arena $(DBG_DIR)/run_arena

$(BUILD_DIR)/run_arena : $(OBJ) run_arena.cpp
	# Create build directories - same structure as sources.
	mkdir -p $(@D)
	# Just link all the object files.
	$(CC) $(CPPFLAGS) -O3 $^ -o $@

$(DBG_DIR)/run_arena : $(DBG_OBJ) run_arena.cpp
	# Create build directories - same structure as sources.
	mkdir -p $(@D)
	# Just link all the object files.
	$(CC) $(CPPFLAGS) -ggdb $^ -o $@

-include $(DEP)

# Build target for every single object file.
# The potential dependency on header files is covered
# by calling `-include $(DEP)`.
$(BUILD_DIR)/%.o : %.cpp
	mkdir -p $(@D)
	# The -MMD flags additionaly creates a .d file with
	# the same name as the .o file.
	$(CC) $(CPPFLAGS) -O3 -MMD -c $< -o $@

$(DBG_DIR)/%.o : %.cpp
	mkdir -p $(@D)
	# The -MMD flags additionaly creates a .d file with
	# the same name as the .o file.
	$(CC) $(CPPFLAGS) -ggdb -MMD -c $< -o $@

.PHONY : clean

clear_data:
	rm -rf data/* save/* brains/*

clean :
	# This should remove all generated files.
	-rm -rf {$(BUILD_DIR),$(DBG_DIR)}/{pure_train,run_arena} $(OBJ) $(DBG_OBJ) $(DEP) pod_codingame.cpp

