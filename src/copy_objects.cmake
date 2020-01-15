# input variabls
# ${OBJECTS} all object files
# ${OUTPUT_DIR} the output directory

function( largest_common_prefix a b prefix )

  # minimum of lengths of both strings
  string( LENGTH ${a} len_a )
  string( LENGTH ${a} len_b )

  if( ${len_a} LESS ${len_b} )
    set( len ${len_a} )
  else()
    set( len ${len_b} )
  endif()

  # iterate over the length
  foreach( end RANGE 1 ${len} )
    # get substrings
    string( SUBSTRING ${a} 0 ${end} sub_a )
    string( SUBSTRING ${b} 0 ${end} sub_b )

    # if equal store, otherwise break
    if ( ${sub_a} STREQUAL ${sub_b} )
      set( ${prefix} ${sub_a} PARENT_SCOPE )
    else()
      break()
    endif()
  endforeach()

endfunction()

function( prefix_dir dirs from_dir )
  set(dirs_num)
  list(LENGTH dirs dirs_num)
  if(${dirs_num} EQUAL 1)
    get_filename_component(dir ${dirs} DIRECTORY)
    set(${from_dir} ${dir} PARENT_SCOPE)
  else()
    set(common "")
    list(GET dirs 0 common)
    foreach(d IN ITEMS ${dirs})
      set(tmp)
      largest_common_prefix(${d} ${common} tmp)
      set(common ${tmp})
    endforeach()
    # if there is no common prefix in filename, ${common} should be of the form
    # <some directory>/ (with a blackslash, but nothing behind)
    # otherwise it will be of the form 
    # <some directory>/<common prefix of filename>
    # add some post fix to ${common} to make sure it always have the latter form,
    # as the <common prefix of filename> will be stripped by the 
    # get_filename_component command
    set(common "${common}p")
    get_filename_component(p ${common} DIRECTORY)
    set(${from_dir} ${p} PARENT_SCOPE)
  endif()
endfunction()

set(FROM_DIR "")
prefix_dir("${OBJECTS}" FROM_DIR)
string(LENGTH ${FROM_DIR} prefix_len)

foreach(obj IN ITEMS ${OBJECTS})
  string(SUBSTRING "${obj}" ${prefix_len} -1 obj_fname)
  set(dest "${OUTPUT_DIR}${obj_fname}")
  get_filename_component(dest_dir ${dest} DIRECTORY)
  file(COPY ${obj} DESTINATION ${dest_dir})
endforeach()
