CC=g++
SOURCES=agent.cpp choice.cpp game_generator.cpp pod_game.cpp pod_game_generator.cpp evaluator.cpp team_evaluator.cpp simple_pod_evaluator.cpp tree_evaluator.cpp arena.cpp game.cpp pod_agent.cpp population_manager.cpp random_tournament.cpp utility.cpp
BUILD_DIR=./build
OBJ = $(SOURCES:%.cpp=$(BUILD_DIR)/%.o)
DEP = $(OBJ:%.o=%.d)

# Disable default rules
.SUFFIXES:

default: pure_train run_arena

pure_train : $(BUILD_DIR)/pure_train pure_train.cpp

$(BUILD_DIR)/pure_train : $(OBJ) pure_train.cpp
	echo "Case 1"
	# Create build directories - same structure as sources.
	mkdir -p $(@D)
	# Just link all the object files.
	$(CC) $(CPPFLAGS) $^ -o $@

run_arena : $(BUILD_DIR)/run_arena

$(BUILD_DIR)/run_arena : $(OBJ) run_arena.cpp
	echo "Case 2"
	# Create build directories - same structure as sources.
	mkdir -p $(@D)
	# Just link all the object files.
	$(CC) $(CPPFLAGS) $^ -o $@


-include $(DEP)

# Build target for every single object file.
# The potential dependency on header files is covered
# by calling `-include $(DEP)`.
$(BUILD_DIR)/%.o : %.cpp
	echo "Case 3"
	mkdir -p $(@D)
	# The -MMD flags additionaly creates a .d file with
	# the same name as the .o file.
	$(CC) $(CPPFLAGS) -MMD -c $< -o $@

.PHONY : clean

clean :
	# This should remove all generated files.
	-rm -rf $(BUILD_DIR)/{pure_train,run_arena} $(OBJ) $(DEP) pod_codingame.cpp

