#include <vector>
#include <unordered_map>
#include <cstdint>
#include <stdio.h>

typedef std::unordered_map<std::uint64_t, std::uint64_t> cEventContext;

struct cEvent {
    std::uint64_t event_type;
    cEventContext event_context;
    std::unordered_map<std::uint64_t, cEventContext>* entity_context;
    void* data;

    cEvent() {
    }

    cEvent(std::uint64_t et, cEventContext ec, std::unordered_map<std::uint64_t, cEventContext>* ent_c, void* d) {
        event_type = et;
        event_context = ec;
        entity_context = ent_c;
        data = d;
    }
};

struct cFlagRef {
    std::uint64_t fact;

    cFlagRef() {
    }

    cFlagRef(std::uint64_t f) {
        fact = f;
    }

    std::uint64_t resolve(cEvent* event, cEventContext* character_context) const {
        //printf("resolving %lu\n", fact);
        auto itr = event->event_context.find(fact);
        if(itr != event->event_context.end()) {
            //printf("found %lu in event context: %lu\n", fact, itr->second);
            return itr->second;
        }
        itr = character_context->find(fact);
        if(itr != character_context->end()) {
            //printf("found %lu in character context: %lu\n", fact, itr->second);
            return itr->second;
        }

        return 0;
    }
};

struct cEntityRef {
    std::uint64_t entity_fact;
    std::uint64_t sub_fact;

    cEntityRef() {
    }

    cEntityRef(std::uint64_t ef, std::uint64_t sf) {
        entity_fact = ef;
        sub_fact = sf;
    }

    std::uint64_t resolve(cEvent* event, cEventContext* character_context) const {
        //printf("resolving $%lu.%lu\n", entity_fact, sub_fact);
        std::uint64_t entity_id;
        auto itr = event->event_context.find(entity_fact);
        if(itr != event->event_context.end()) {
            //printf("found $%lu in event context: %lu\n", entity_fact, itr->second);
            entity_id = itr->second;
        } else if((itr = character_context->find(entity_fact)) != character_context->end()) {
            //printf("found $%lu in character context: %lu\n", entity_fact, itr->second);
            entity_id = itr->second;
        } else {
            return 0;
        }

        // then apply criteria on the fact within that entity
        const auto &eitr = event->entity_context->find(entity_id);
        if(eitr == event->entity_context->end()) {
            return 0;
        }
        cEventContext &entity_context = eitr->second;
        const auto &sfitr = entity_context.find(sub_fact);
        if(sfitr != entity_context.end()) {
            //printf("found $%lu.%lu in entity context: %lu\n", entity_fact, sub_fact, sfitr->second);
            return sfitr->second;
        }
        return 0;
    }
};

template<class T>
struct cCriteria{
    T fact;
    std::uint64_t low;
    std::uint64_t high;

    cCriteria() {
    }

    cCriteria(T f, std::uint64_t l, std::uint64_t h) {
        fact = f;
        low = l;
        high = h;
    }

    bool evaluate(cEvent* event, cEventContext* character_context) const {
        std::uint64_t fact_value = fact.resolve(event, character_context);
        //printf("comparing %lu <= %lu <= %lu\n", low, fact_value, high);
        return low <= fact_value && fact_value <= high;
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

struct cAction {
    std::uint64_t action_id;
    cCharacterCandidate character_candidate;
    void* args;

    cAction() {
    }

    cAction(cCharacterCandidate c, std::uint64_t aid, void* a) {
        character_candidate = c;
        action_id = aid;
        args = a;
    }
};

struct cActionTemplate {
    std::uint64_t action_id;
    void* args;

    cActionTemplate() {
    }

    cActionTemplate(std::uint64_t aid, void* a) {
        action_id = aid;
        args = a;
    }

    cAction resolve(cEvent *event, cCharacterCandidate character_candidate) const {
        return cAction(character_candidate, action_id, args);
    }
};

class cRule {
    private:
        std::uint64_t event_type;
        std::uint64_t priority;
        std::vector<cCriteria<cFlagRef> > criteria;
        std::vector<cCriteria<cEntityRef> > entity_criteria;
        std::vector<cActionTemplate> actions;

    public:
        cRule() {
        }

        cRule(
            std::uint64_t et,
            std::uint64_t pri,
            std::vector<cCriteria<cFlagRef> > cri,
            std::vector<cCriteria<cEntityRef> > e_cri,
            std::vector<cActionTemplate> a
        ) {
            event_type = et;
            priority = pri;
            criteria = cri;
            entity_criteria = e_cri;
            actions = a;
        }

        const std::vector<cActionTemplate>& get_actions() const {
            return actions;
        }

        bool evaluate(cEvent* event, cEventContext* character_context) const {
            //printf("evaluating rule criteria\n");
            for(auto &c : criteria) {
                if(!c.evaluate(event, character_context)) {
                    //printf("failed criteria %lu\n", c.fact.fact);
                    return false;
                }
            }

            //printf("evaluating rule entity criteria\n");
            for(auto &ec : entity_criteria) {
                if(!ec.evaluate(event, character_context)) {
                    //printf("failed entity criteria $%lu.%lu\n", ec.fact.entity_fact, ec.fact.sub_fact);
                    return false;
                }
            }

            //printf("passed all criteria\n");
            return true;
        }
};

class cDirector {
    private:
        // stored in descending priority sorted order
        std::unordered_map<std::uint64_t, std::vector<cRule> > rules;

    public:
        cDirector() {
        }

        cDirector(std::unordered_map<std::uint64_t, std::vector<cRule> > r) {
            rules = r;
        }

        std::vector<cAction> evaluate(cEvent* event, std::vector<cCharacterCandidate> character_candidates) const {
            std::vector<cAction> matches;

            // find the rules for this event id
            const auto &ritr = rules.find(event->event_type);
            if(ritr == rules.end()) {
                return matches;
            }
            const std::vector<cRule> &matching_rules = ritr->second;

            // find the "best" matching rule for each character
            // best here means first matching rule in descending priority order
            for(auto &character_candidate : character_candidates) {
                //TODO: what do we do for matches in equal priority? first wins
                for(const auto &itr : matching_rules) {
                    if(itr.evaluate(event, character_candidate.character_context)) {
                        for(const auto &aitr : itr.get_actions()) {
                            //printf("adding action %lu\n", aitr.action_id);
                            matches.push_back(aitr.resolve(event, character_candidate));
                        }
                        break;
                    }
                }
            }
            return matches;
        }
};
