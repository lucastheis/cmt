#include "exception.h"
#include "preconditioner.h"
#include "pcapreconditioner.h"
#include "conditionaldistribution.h"

template <class CD, class PC = CMT::PCAPreconditioner>
class PreconditionedModel : public ConditionalDistribution {
	public:
		typedef typename CD::Parameters Parameters;

		PreconditionedModel(CD* cd, PC* pc = 0);
		virtual ~PreconditionedModel();

		inline const PC* preconditioner() const;
		inline void setPreconditioner(PC* pc);

		virtual MatrixXd sample(const MatrixXd& input) const;
		virtual Array<double, 1, Dynamic> logLikelihood(
			const MatrixXd& input,
			const MatrixXd& output) const;

		virtual void initialize(
			const MatrixXd& input,
			const MatrixXd& output);
		virtual bool train(
			const MatrixXd& input,
			const MatrixXd& output,
			const Parameters& params = Parameters());
		virtual bool train(
			const MatrixXd& input,
			const MatrixXd& output,
			const MatrixXd& inputVal,
			const MatrixXd& outputVal,
			const Parameters& params = Parameters());

	protected:
		CD* mCD;
		PC* mPC;
		bool mOwner;
};



template <class CD, class PC>
PreconditionedModel<CD, PC>::PreconditionedModel(CD* cd, PC* pc) : mCD(cd), mPC(pc) {
	if(mPC) {
		mOwner = false;
		if(mPC->dimInPre() != mCD->dimIn() || mPC->dimOutPre() != mCD->dimOut())
			throw Exception("Model and preconditioner are incompatible.");
	} else {
		mOwner = true;
	}
}



template <class CD, class PC>
PreconditionedModel<CD, PC>::~PreconditionedModel() {
	if(mOwner && mPC)
		delete mPC;
}



template <class CD, class PC>
const PC* PreconditionedModel<CD, PC>::preconditioner() const {
	return mPC;
}



template <class CD, class PC>
void PreconditionedModel<CD, PC>::setPreconditioner(PC* pc) {
	if(mOwner && mPC)
		delete mPC;
	mPC = pc;
	mOwner = false;
}



template <class CD, class PC>
MatrixXd PreconditionedModel<CD, PC>::sample(const MatrixXd& input) const {
	if(!mPC)
		throw Exception("Model has to be initialized first.");
	MatrixXd inputPc = mPC->operator()(input);
	MatrixXd outputPc = mCD->sample(inputPc);
	return mPC->inverse(inputPc, outputPc).second;
}



template <class CD, class PC>
Array<double, 1, Dynamic> PreconditionedModel<CD, PC>::logLikelihood(
	const MatrixXd& input,
	const MatrixXd& output) const
{
	if(!mPC)
		throw Exception("Model has to be initialized first.");
	return 
		mCD->logLikelihood(mPC->operator()(input, output)) + 
		mPC->logJacobian(input, output);
}



template <class CD, class PC>
void PreconditionedModel<CD, PC>::initialize(
	const MatrixXd& input,
	const MatrixXd& output)
{
	if(!mPC)
		mPC = new PC(input, output);
	return mCD->initialize(mPC->operator()(input, output));
}



template <class CD, class PC>
bool PreconditionedModel<CD, PC>::train(
	const MatrixXd& input,
	const MatrixXd& output,
	const Parameters& params)
{
	if(!mPC)
		throw Exception("Model has to be initialized first.");
	return mCD->train(mPC->operator()(input, output), params);
}



template <class CD, class PC>
bool PreconditionedModel<CD, PC>::train(
	const MatrixXd& input,
	const MatrixXd& output,
	const MatrixXd& inputVal,
	const MatrixXd& outputVal,
	const Parameters& params)
{
	if(!mPC)
		throw Exception("Model has to be initialized first.");
	return mCD->train(
		mPC->operator()(input, output),
		mPC->operator()(inputVal, outputVal),
		params);
}
