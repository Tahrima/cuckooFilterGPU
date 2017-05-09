/**
 * \file sha1.h
 */
#ifndef _SHA1_H
#define _SHA1_H

/**
 * \brief          SHA-1 context structure
 */
typedef struct {
	unsigned long total[2]; /*!< number of bytes processed  */
	unsigned long state[5]; /*!< intermediate digest state  */
	unsigned char buffer[64]; /*!< data block being processed */
} sha1_context;

/**
 * \brief          SHA-1 context setup
 *
 * \param ctx      context to be initialized
 */
__device__
void sha1_starts(sha1_context *ctx);

/**
 * \brief          SHA-1 process buffer
 *
 * \param ctx      SHA-1 context
 * \param input    buffer holding the  data
 * \param ilen     length of the input data
 */
__device__
void sha1_update(sha1_context *ctx, unsigned char *input, int ilen);

/**
 * \brief          SHA-1 final digest
 *
 * \param ctx      SHA-1 context
 * \param output   SHA-1 checksum result
 */
__device__
void sha1_finish(sha1_context *ctx, unsigned char *output);

/**
 * \brief          Output = SHA-1( input buffer )
 *
 * \param input    buffer holding the  data
 * \param ilen     length of the input data
 * \param output   SHA-1 checksum result
 */
__device__
void sha1(unsigned char *input, int ilen, unsigned char *output);

#endif /* sha1.h */
