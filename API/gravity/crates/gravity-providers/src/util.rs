//! Small shared helpers for provider implementations.

use gravity_core::Env;

/// Generate an RFC-4122 v4 UUID string from injected [`Env`] entropy.
///
/// Using `env.entropy` (rather than the OS RNG directly) keeps request bodies
/// that embed UUIDs reproducible under golden-trace tests.
pub fn uuid_v4(env: &Env) -> String {
    let mut b = [0u8; 16];
    env.entropy.fill(&mut b);
    // Set version (4) and variant (RFC 4122) bits.
    b[6] = (b[6] & 0x0F) | 0x40;
    b[8] = (b[8] & 0x3F) | 0x80;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuid_has_v4_shape_and_is_deterministic() {
        let env = Env::fixed(0, 7);
        let u = uuid_v4(&env);
        assert_eq!(u.len(), 36);
        assert_eq!(u.as_bytes()[14], b'4', "version nibble");
        assert!(matches!(u.as_bytes()[19], b'8' | b'9' | b'a' | b'b'), "variant nibble");
        // same seed → same uuid
        assert_eq!(uuid_v4(&Env::fixed(0, 7)), u);
    }
}
