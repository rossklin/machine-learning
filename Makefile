CC=g++
CPPFLAGS=--std=c++17 -O3 -fopenmp
SOURCES=agent.cpp choice.cpp game_generator.cpp pod_game.cpp pod_game_generator.cpp simple_pod_evaluator.cpp arena.cpp game.cpp pod_agent.cpp population_manager.cpp random_tournament.cpp utility.cpp
BIN1 = pure_train
BIN2 = run_arena
BUILD_DIR=./build
OBJ = $(SOURCES:%.cpp=$(BUILD_DIR)/%.o)
DEP = $(OBJ:%.o=%.d)

$(BIN1) : $(BUILD_DIR)/$(BIN1)

$(BUILD_DIR)/$(BIN1) : $(OBJ) $(BIN1).cpp
	# Create build directories - same structure as sources.
	mkdir -p $(@D)
	# Just link all the object files.
	$(CC) $(CPPFLAGS) $^ -o $@

$(BIN2) : $(BUILD_DIR)/$(BIN2)

$(BUILD_DIR)/$(BIN2) : $(OBJ) $(BIN2).cpp
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

