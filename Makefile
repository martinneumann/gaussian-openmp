TARGET ?= blur
SRC_DIRS ?=  .                                                              

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s)          
OBJS := $(addsuffix .o,$(basename $(SRCS)))                                     
DEPS := $(OBJS:.o=.d)                                                           

INC_DIRS := $(shell find $(SRC_DIRS) -type d)                                   
INC_FLAGS := $(addprefix -I,$(INC_DIRS))                                        

LDLIBS := `pkg-config --cflags --libs opencv`

CPPFLAGS ?= $(INC_FLAGS) -MMD -MP

$(TARGET): $(OBJS)                                                              
	g++ $(LDFLAGS) $(OBJS) -o $@ $(LOADLIBES) $(LDLIBS) -fopenmp

.PHONY: clean                                                                   
	clean:                                                                      
	$(RM) $(TARGET) $(OBJS) $(DEPS)                                         

-include $(DEPS)                                                                

                          
