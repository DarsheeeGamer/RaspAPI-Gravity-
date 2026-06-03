/*
 * gravity.h — stable C ABI for the Gravity AI gateway (pure-Rust core).
 *
 * All char* returns are heap-allocated UTF-8 and MUST be released with
 * gravity_free_string(), EXCEPT gravity_version() which returns a static
 * pointer. The request/response boundary is JSON.
 */
#ifndef GRAVITY_H
#define GRAVITY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/* Lifecycle */
int         gravity_init(void);
const char* gravity_version(void);            /* static — do NOT free */
void        gravity_free_string(char* ptr);

/* Blocking chat. request_json -> response_json (caller frees). */
char* gravity_chat(const char* request_json);

/* Embeddings. request_json -> response_json (caller frees). */
char* gravity_embeddings(const char* request_json);

/* Provider discovery -> JSON array (caller frees). */
char* gravity_list_providers(void);

/* Callback streaming. Blocks; callback invoked per token (token valid only
 * during the call). Returns 0 on success, -1 on error. */
typedef void (*gravity_token_cb)(const char* token, void* userdata);
int gravity_chat_stream(const char* request_json,
                        gravity_token_cb callback,
                        void* userdata);

/* Pull streaming. open -> opaque handle (NULL on error); next -> token string
 * (caller frees) or NULL at end of stream; close -> release handle. */
void* gravity_stream_open(const char* request_json);
char* gravity_stream_next(void* handle);
void  gravity_stream_close(void* handle);

#ifdef __cplusplus
}
#endif

#endif /* GRAVITY_H */
