#ifndef NARRATIVE_DIRECTOR_H
#define NARRATIVE_DIRECTOR_H

#include <vector>
#include <unordered_map>
#include <cstdint>
#include <stdio.h>
#include <memory>

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

/*class cRef {
    public:
    virtual ~cRef() {}
    virtual std::uint64_t resolve(cEvent* event, cEventContext* character_context) const = 0;
};*/

struct cIntRef {
    std::uint64_t value;

    cIntRef() {
    }

    cIntRef(std::uint64_t v) {
        value = v;
    }

    std::uint64_t resolve(cEvent* event, cEventContext* character_context) const {
        return value;
    }

    std::uint64_t key() const {
        return 0;
    }
};

struct cFlagRef {
    std::uint64_t fact;

    cFlagRef() : fact(0) {
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

    std::uint64_t key() const {
        return fact;
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

    std::uint64_t key() const {
        return sub_fact;
    }
};

struct cCriteriaBase {
    cCriteriaBase() {}
    virtual ~cCriteriaBase() {}
    virtual bool evaluate(cEvent* event, cEventContext* character_context) const = 0;
    virtual std::uint64_t distance(cEvent* event, cEventContext* character_context) const = 0;
    virtual std::uint64_t key() const = 0;
    virtual std::unique_ptr<cCriteriaBase> clone() const = 0;
};

template<class L, class F, class U>
struct cCriteria : cCriteriaBase {
    F fact;
    L low;
    U high;

    cCriteria() {
    }

    cCriteria(L l, F f, U h) {
        fact = f;
        low = l;
        high = h;
    }

    virtual bool evaluate(cEvent* event, cEventContext* character_context) const {
        std::uint64_t fact_value = fact.resolve(event, character_context);
        std::uint64_t low_value = low.resolve(event, character_context);
        std::uint64_t high_value = high.resolve(event, character_context);
        //printf("comparing %lu <= %lu <= %lu\n", low_value, fact_value, high_value);
        return low_value <= fact_value && fact_value <= high_value;
    }

    virtual std::uint64_t distance(cEvent* event, cEventContext* character_context) const {
        std::uint64_t fact_value = fact.resolve(event, character_context);
        std::uint64_t low_value = low.resolve(event, character_context);
        std::uint64_t high_value = high.resolve(event, character_context);

        if(fact_value < low_value) {
            return low_value - fact_value;
        } else if (fact_value > high_value) {
            return fact_value - high_value;
        } else {
            return 0;
        }
    }

    virtual std::uint64_t key() const {
        return fact.key();
    }

    virtual std::unique_ptr<cCriteriaBase> clone() const {
        return std::make_unique<cCriteria<L, F, U>>(low, fact, high);
    }

    operator bool() const {
        return key() > 0;
    }
};


struct UBuilder {
    virtual ~UBuilder() {}
    virtual std::unique_ptr<cCriteriaBase> addU(cIntRef u) = 0;
    virtual std::unique_ptr<cCriteriaBase> addU(cFlagRef u) = 0;
    virtual std::unique_ptr<cCriteriaBase> addU(cEntityRef u) = 0;
};

struct FBuilder {
    virtual ~FBuilder() {}
    virtual std::unique_ptr<UBuilder> addF(cIntRef f) = 0;
    virtual std::unique_ptr<UBuilder> addF(cFlagRef f) = 0;
    virtual std::unique_ptr<UBuilder> addF(cEntityRef f) = 0;
};

template<class L, class F>
struct UBuilderImpl : UBuilder {
    L l;
    F f;
    UBuilderImpl(L low, F fact) {
        l = low;
        f = fact;
    }
    virtual std::unique_ptr<cCriteriaBase> addU(cIntRef u) {
        return std::make_unique<cCriteria<L, F, cIntRef> >(l, f, u);
    }
    virtual std::unique_ptr<cCriteriaBase> addU(cFlagRef u) {
        return std::make_unique<cCriteria<L, F, cFlagRef> >(l, f, u);
    }
    virtual std::unique_ptr<cCriteriaBase> addU(cEntityRef u) {
        return std::make_unique<cCriteria<L, F, cEntityRef> >(l, f, u);
    }
};

template<class L>
struct FBuilderImpl : FBuilder {
    L l;
    FBuilderImpl(L low) {
        l = low;
    }
    virtual std::unique_ptr<UBuilder> addF(cIntRef f) {
        return std::make_unique<UBuilderImpl<L, cIntRef> >(l, f);
    }
    virtual std::unique_ptr<UBuilder> addF(cFlagRef f) {
        return std::make_unique<UBuilderImpl<L, cFlagRef> >(l, f);
    }
    virtual std::unique_ptr<UBuilder> addF(cEntityRef f) {
        return std::make_unique<UBuilderImpl<L, cEntityRef> >(l, f);
    }
};

struct cCriteriaBuilder {
    std::unique_ptr<FBuilder> fbuilder;
    std::unique_ptr<UBuilder> ubuilder;
    std::unique_ptr<cCriteriaBase> criteria;

    cCriteriaBuilder() {
    }

    template<class L>
    void addL(L l) {
        fbuilder = std::make_unique<FBuilderImpl<L> >(l);
    }

    template<class F>
    void addF(F f) {
        ubuilder = fbuilder->addF(f);
    }

    template<class U>
    void addU(U u) {
        criteria = ubuilder->addU(u);
    }

    std::unique_ptr<cCriteriaBase> build() {
        return std::move(criteria);
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
        std::vector<std::unique_ptr<cCriteriaBase>> criteria;
        std::vector<cActionTemplate> actions;

    public:
        cRule() {
        }

        cRule(
            std::uint64_t et,
            std::uint64_t pri,
            std::vector<std::unique_ptr<cCriteriaBase>> &cri,
            std::vector<cActionTemplate> a
        ) {
            event_type = et;
            priority = pri;
            actions = a;

            for(auto &c : cri) {
                criteria.push_back(std::move(c));
            }
        }

        const std::vector<cActionTemplate>& get_actions() const {
            return actions;
        }

        bool evaluate(cEvent* event, cEventContext* character_context) const {
            //printf("evaluating rule criteria\n");
            for(auto &c : criteria) {
                if(!c->evaluate(event, character_context)) {
                    //printf("failed criteria %lu\n", c.fact.fact);
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
        std::unordered_map<std::uint64_t, std::vector<std::unique_ptr<cRule>> > rules;

    public:
        cDirector() {
        }

        cDirector(std::unordered_map<std::uint64_t, std::vector<std::unique_ptr<cRule>> > &r) {
            for(auto &entry : r) {
                rules[entry.first] = std::vector<std::unique_ptr<cRule> >();
                for(auto &one_r : entry.second) {
                    rules[entry.first].push_back(std::move(one_r));
                }
            }
        }

        std::vector<cAction> evaluate(cEvent* event, std::vector<cCharacterCandidate> character_candidates) const {
            std::vector<cAction> matches;

            // find the rules for this event id
            const auto &ritr = rules.find(event->event_type);
            if(ritr == rules.end()) {
                return matches;
            }
            const std::vector<std::unique_ptr<cRule>> &matching_rules = ritr->second;

            // find the "best" matching rule for each character
            // best here means first matching rule in descending priority order
            for(auto &character_candidate : character_candidates) {
                //TODO: what do we do for matches in equal priority? first wins
                for(const auto &itr : matching_rules) {
                    if(itr->evaluate(event, character_candidate.character_context)) {
                        for(const auto &aitr : itr->get_actions()) {
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

#endif /* NARRATIVE_DIRECTOR_H */
