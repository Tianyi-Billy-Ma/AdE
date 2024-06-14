from .model import model as AdE

def parse_model(args, data):
    print(f"Method: {args.method}")
    if args.method == "AdE":
        model = AdE(V=data.x.shape[0],
                      X=data.x,
                      E=data.edge_index.T,
                      num_features=args.num_features,
                      num_layers=args.num_layers,
                      num_classes=args.num_classes,
                      args=args
                      )
    return model