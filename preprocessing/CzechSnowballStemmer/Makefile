snowball_sources= \
  src_c/stem_UTF_8_english.c \
  src_c/stem_UTF_8_german.c \
  src_c/stem_UTF_8_czech.c \
  runtime/api.c \
  runtime/utilities.c \
  libstemmer/libstemmer.c

snowball_headers= \
  src_c/stem_UTF_8_english.h \
  src_c/stem_UTF_8_german.h \
  src_c/stem_UTF_8_czech.h \
  libstemmer/libstemmer.h \
  libstemmer/modules.h \
  runtime/api.h \
  runtime/header.h
  
#CFLAGS=-Iinclude
all: libstemmer.o stemmer
libstemmer.o: $(snowball_sources:.c=.o)
	$(AR) -cru $@ $^
stemmer: examples/stemwords.o libstemmer.o
	$(CC) -o $@ $^
clean:
	rm -f stemmer *.o src_c/*.o runtime/*.o libstemmer/*.o
