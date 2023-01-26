#include <vector>
#include <unordered_map>
#include <cstdint>

typedef std::unordered_map<std::uint64_t, std::uint64_t> cEventContext;

struct cEvent {
    std::uint64_t event_type;
    cEventContext* event_context;
    std::unordered_map<std::uint64_t, cEventContext*> entity_context;

    cEvent() {
    }
};

class cFlagCriteria{
    private:
        std::uint64_t fact;
        std::uint64_t low;
        std::uint64_t high;

    public:
        cFlagCriteria() {
        }

        cFlagCriteria(std::uint64_t f, std::uint64_t l, std::uint64_t h) {
            fact = f;
            low = l;
            high = h;
        }

        bool evaluate(cEvent* event, cEventContext* character_context) {
            auto itr = event->event_context->find(fact);
            if(itr != event->event_context->end()) {
                return low <= itr->second && itr->second <= high;
            }
            itr = character_context->find(fact);
            if(itr != character_context->end()) {
                return low <= itr->second && itr->second <= high;
            }
            return false;
        }
};

class cEntityCriteria {
    private:
        std::uint64_t entity_fact;
        std::uint64_t sub_fact;
        std::uint64_t low;
        std::uint64_t high;

    public:
        cEntityCriteria() {
        }

        cEntityCriteria(std::uint64_t ef, std::uint64_t sf, std::uint64_t l, std::uint64_t h) {
            entity_fact = ef;
            sub_fact = sf;
            low = l;
            high = h;
        }

        bool evaluate(cEvent* event, cEventContext* character_context) {
            // first find entity_id from event or character contexts
            std::uint64_t entity_id;
            auto itr = event->event_context->find(entity_fact);
            if(itr != event->event_context->end()) {
                entity_id = itr->second;
            } else if((itr = character_context->find(entity_fact)) != character_context->end()) {
                entity_id = itr->second;
            } else {
                return false;
            }

            // then apply criteria on the fact within that entity
            auto eitr = event->entity_context.find(entity_id);
            if(eitr == event->entity_context.end()) {
                //TODO: is this actually an error?
                return false;
            }
            cEventContext *entity_context = eitr->second;
            auto sfitr = entity_context->find(sub_fact);
            if(sfitr != entity_context->end()) {
                return low <= sfitr->second && sfitr->second <= high;
            }
            return false;
        }
};

class cRule {
    private:
        std::uint64_t rule_id;
        std::uint64_t event_type;
        std::uint64_t priority;
        std::vector<cFlagCriteria> criteria;
        std::vector<cEntityCriteria> entity_criteria;

    public:
        cRule() {
        }

        cRule(
            std::uint64_t rid,
            std::uint64_t et,
            std::uint64_t pri,
            std::vector<cFlagCriteria> cri,
            std::vector<cEntityCriteria> e_cri
        ) {
            rule_id = rid;
            event_type = et;
            priority = pri;
            criteria = cri;
            entity_criteria = e_cri;
        }

        std::uint64_t get_rule_id() {
            return rule_id;
        }

        bool evaluate(cEvent* event, cEventContext* character_context) {
            for(auto &c : criteria) {
                if(!c.evaluate(event, character_context)) {
                    return false;
                }
            }

            for(auto &ec : entity_criteria) {
                if(!ec.evaluate(event, character_context)) {
                    return false;
                }
            }

            return true;
        }
};

struct cCharacterCandidate {
    cEventContext* character_context;
    void* data;

    cCharacterCandidate() {
    }

    cCharacterCandidate(cEventContext* cc, void* d) {
        character_context = cc;
        data = d;
    }

};

struct cRuleMatch {
    std::uint64_t rule_id;
    cCharacterCandidate character_candidate;

    cRuleMatch(std::uint64_t rid, cCharacterCandidate cc) {
        rule_id = rid;
        character_candidate = cc;
    }
};

class cDirector {
    private:
        // stored in descending priority sorted order
        std::unordered_map<std::uint64_t, std::vector<cRule> > rules;

    public:
        std::vector<cRuleMatch> evaluate(cEvent* event, std::vector<cCharacterCandidate> character_candidates) {
            std::vector<cRuleMatch> matches;

            // find the rules for this event id
            auto ritr = rules.find(event->event_type);
            if(ritr == rules.end()) {
                return matches;
            }
            std::vector<cRule> &matching_rules = ritr->second;

            // find the "best" matching rule for each character
            // best here means first matching rule in descending priority order
            for(auto &character_candidate : character_candidates) {
                //TODO: what do we do for matches in equal priority? first wins
                const auto &itr = matching_rules.begin();
                while(itr != matching_rules.end()) {
                    if(itr->evaluate(event, character_candidate.character_context)) {
                        matches.push_back(cRuleMatch(itr->get_rule_id(), character_candidate));
                        break;
                    }
                }
            }
            return matches;
        }
};
