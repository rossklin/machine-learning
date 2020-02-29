CC=g++
CPPFLAGS=--std=c++17 -O3 -fopenmp
SOURCES=agent.cpp choice.cpp game_generator.cpp pod_game.cpp pure_train.cpp run_arena.cpp simple_pod_evaluator.cpp arena.cpp game.cpp pod_agent.cpp population_manager.cpp random_tournament.cpp run_local.cpp utility.cpp
BIN = pure_train
BUILD_DIR=./build
OBJ = $(SOURCES:%.cpp=$(BUILD_DIR)/%.o)
DEP = $(OBJ:%.o=%.d)

$(BIN) : $(BUILD_DIR)/$(BIN)

$(BUILD_DIR)/$(BIN) : $(OBJ)
	# Create build directories - same structure as sources.
	mkdir -p $(@D)
	# Just link all the object files.
	$(CC) $(CPPFLAGS) $^ -o $@

-include $(DEP)

# Build target for every single object file.
# The potential dependency on header files is covered
# by calling `-include $(DEP)`.
$(BUILD_DIR)/%.o : %.cpp
	mkdir -p $(@D)
	# The -MMD flags additionaly creates a .d file with
	# the same name as the .o file.
	$(CC) $(CPPFLAGS) -MMD -c $< -o $@

.PHONY : clean

clean :
	# This should remove all generated files.
	-rm -rf $(BUILD_DIR)/$(BIN) $(OBJ) $(DEP) brains *.csv pod_codingame.cpp
	mkdir brains

