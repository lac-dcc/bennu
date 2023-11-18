#!/bin/bash

if [ -z "$TVM_HOME" ]
then
    echo "\$TVM_HOME is empty, please setup the TVM variable"
else
    # copy files to TVM project
    cp src/run_state.* $TVM_HOME/src/auto_scheduler/

    # Go to TVM project and compile it
    cd $TVM_HOME/build
    cmake ..
    make -j16
    # Go back to the project
    cd -
fi

